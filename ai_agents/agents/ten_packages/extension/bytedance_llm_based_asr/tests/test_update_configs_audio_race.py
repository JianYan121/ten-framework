import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from ..config import BytedanceASRLLMConfig
from ..extension import BytedanceASRLLMExtension


def _minimal_config() -> BytedanceASRLLMConfig:
    return BytedanceASRLLMConfig.model_validate(
        {
            "params": {
                "audio": {"rate": 16000},
                "request": {"model_name": "bigmodel"},
            }
        }
    )


class _RaceClient:
    def __init__(self) -> None:
        self.connected = True
        self.audio_send_started = asyncio.Event()
        self.allow_audio_send = asyncio.Event()
        self.finalize_started = asyncio.Event()
        self.allow_finalize = asyncio.Event()
        self.sent_audio: list[bytes] = []
        self.finalize_calls = 0

    async def send_audio(self, audio_data: bytes) -> None:
        self.audio_send_started.set()
        await self.allow_audio_send.wait()
        if not self.connected:
            raise RuntimeError("Not connected to ASR service")
        self.sent_audio.append(audio_data)

    async def finalize(self) -> None:
        self.finalize_started.set()
        await self.allow_finalize.wait()
        if not self.connected:
            raise RuntimeError("Not connected to ASR service")
        self.finalize_calls += 1


def _new_extension(client: object) -> BytedanceASRLLMExtension:
    extension = BytedanceASRLLMExtension("test_extension")
    extension.ten_env = MagicMock()
    extension.config = _minimal_config()
    extension.connected = True
    extension.client = client
    extension.send_asr_error = AsyncMock()
    return extension


def _frame() -> MagicMock:
    frame = MagicMock()
    frame.lock_buf.return_value = b"\x00\x01"
    return frame


def _install_reconnect_mocks(
    extension: BytedanceASRLLMExtension,
    old_client: object,
    new_client: object | None = None,
) -> tuple[AsyncMock, AsyncMock]:
    async def stop_connection() -> None:
        old_client.connected = False
        extension.connected = False
        extension.client = None

    async def start_connection() -> None:
        target_client = new_client or old_client
        target_client.connected = True
        extension.client = target_client
        extension.connected = True

    stop_mock = AsyncMock(side_effect=stop_connection)
    start_mock = AsyncMock(side_effect=start_connection)
    extension.stop_connection = stop_mock
    extension.start_connection = start_mock
    return stop_mock, start_mock


@pytest.mark.asyncio
async def test_update_configs_waits_for_in_flight_audio_before_reconnecting():
    client = _RaceClient()
    extension = _new_extension(client)
    stop_mock, _ = _install_reconnect_mocks(extension, client)

    audio_task = asyncio.create_task(
        extension.send_audio(_frame(), "session-1")
    )
    await client.audio_send_started.wait()
    update_task = asyncio.create_task(
        extension._run_update_configs(
            {
                "params": {
                    "request": {
                        "corpus": {"context": "updated dialog context"},
                    }
                }
            }
        )
    )
    await asyncio.sleep(0)
    stop_mock.assert_not_awaited()
    assert "corpus" not in extension.config.params["request"]
    client.allow_audio_send.set()

    update_result, audio_result = await asyncio.wait_for(
        asyncio.gather(update_task, audio_task),
        timeout=1,
    )

    assert update_result == (True, "")
    assert audio_result is True
    assert extension.is_connected() is True
    assert client.sent_audio == [b"\x00\x01"]
    assert extension.config.params["request"]["corpus"]["context"] == (
        "updated dialog context"
    )
    extension.send_asr_error.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_configs_waits_for_finalize_before_reconnecting():
    client = _RaceClient()
    extension = _new_extension(client)
    stop_mock, _ = _install_reconnect_mocks(extension, client)

    finalize_task = asyncio.create_task(extension.finalize("session-1"))
    await client.finalize_started.wait()
    update_task = asyncio.create_task(
        extension._run_update_configs(
            {"params": {"request": {"enable_nonstream": False}}}
        )
    )
    await asyncio.sleep(0)
    stop_mock.assert_not_awaited()
    client.allow_finalize.set()

    update_result, _ = await asyncio.wait_for(
        asyncio.gather(update_task, finalize_task),
        timeout=1,
    )

    assert update_result == (True, "")
    assert client.finalize_calls == 1
    extension.send_asr_error.assert_not_awaited()


@pytest.mark.asyncio
async def test_automatic_reconnect_waits_for_in_flight_audio():
    client = _RaceClient()
    extension = _new_extension(client)
    extension.min_retry_delay = 0
    stop_mock, _ = _install_reconnect_mocks(extension, client)

    audio_task = asyncio.create_task(
        extension.send_audio(_frame(), "session-1")
    )
    await client.audio_send_started.wait()
    reconnect_task = asyncio.create_task(extension._handle_reconnect())
    await asyncio.sleep(0)
    stop_mock.assert_not_awaited()
    client.allow_audio_send.set()

    audio_result, _ = await asyncio.wait_for(
        asyncio.gather(audio_task, reconnect_task),
        timeout=1,
    )

    assert audio_result is True
    stop_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_audio_waits_for_update_and_uses_reconnected_client():
    old_client = MagicMock()
    old_client.connected = True
    old_client.send_audio = AsyncMock()
    new_client = MagicMock()
    new_client.connected = True
    new_client.send_audio = AsyncMock()
    extension = _new_extension(old_client)
    restart_started = asyncio.Event()
    allow_restart = asyncio.Event()

    async def stop_connection() -> None:
        restart_started.set()
        await allow_restart.wait()
        old_client.connected = False
        extension.connected = False
        extension.client = None

    async def start_connection() -> None:
        extension.client = new_client
        extension.connected = True

    extension.stop_connection = AsyncMock(side_effect=stop_connection)
    extension.start_connection = AsyncMock(side_effect=start_connection)

    update_task = asyncio.create_task(
        extension._run_update_configs(
            {"params": {"request": {"enable_nonstream": False}}}
        )
    )
    await restart_started.wait()
    audio_task = asyncio.create_task(
        extension.send_audio(_frame(), "session-1")
    )
    await asyncio.sleep(0)
    old_client.send_audio.assert_not_awaited()
    new_client.send_audio.assert_not_awaited()
    allow_restart.set()

    update_result, audio_result = await asyncio.wait_for(
        asyncio.gather(update_task, audio_task),
        timeout=1,
    )

    assert update_result == (True, "")
    assert audio_result is True
    old_client.send_audio.assert_not_awaited()
    new_client.send_audio.assert_awaited_once_with(b"\x00\x01")


@pytest.mark.asyncio
async def test_send_audio_error_can_reconnect_without_deadlock():
    failed_client = MagicMock()
    failed_client.connected = True
    failed_client.send_audio = AsyncMock(
        side_effect=RuntimeError("send failed")
    )
    replacement_client = MagicMock()
    replacement_client.connected = True
    extension = _new_extension(failed_client)
    stop_mock, start_mock = _install_reconnect_mocks(
        extension,
        failed_client,
        replacement_client,
    )

    async def handle_error(_error: Exception) -> None:
        await extension._restart_connection()

    extension._handle_error = AsyncMock(side_effect=handle_error)

    result = await asyncio.wait_for(
        extension.send_audio(_frame(), "session-1"),
        timeout=1,
    )

    assert result is False
    stop_mock.assert_awaited_once()
    start_mock.assert_awaited_once()
    assert extension.client is replacement_client


@pytest.mark.asyncio
async def test_failed_audio_send_does_not_update_timeline_or_dumpers():
    client = MagicMock()
    client.connected = True
    client.send_audio = AsyncMock(side_effect=RuntimeError("send failed"))
    extension = _new_extension(client)
    extension._handle_error = AsyncMock()

    audio_dumper = MagicMock()
    audio_dumper.push_bytes = AsyncMock()
    extension.audio_dumper = audio_dumper
    log_id_dumper_manager = MagicMock()
    log_id_dumper_manager.push_bytes = AsyncMock()
    extension.log_id_dumper_manager = log_id_dumper_manager

    result = await extension.send_audio(_frame(), "session-1")

    assert result is False
    assert extension.audio_timeline.get_total_user_audio_duration() == 0
    audio_dumper.push_bytes.assert_not_awaited()
    log_id_dumper_manager.push_bytes.assert_not_awaited()


@pytest.mark.asyncio
async def test_finalize_error_can_reconnect_without_deadlock():
    failed_client = MagicMock()
    failed_client.connected = True
    failed_client.finalize = AsyncMock(
        side_effect=RuntimeError("finalize failed")
    )
    replacement_client = MagicMock()
    replacement_client.connected = True
    extension = _new_extension(failed_client)
    stop_mock, start_mock = _install_reconnect_mocks(
        extension,
        failed_client,
        replacement_client,
    )

    async def handle_error(_error: Exception) -> None:
        await extension._restart_connection()

    extension._handle_error = AsyncMock(side_effect=handle_error)

    await asyncio.wait_for(extension.finalize("session-1"), timeout=1)

    stop_mock.assert_awaited_once()
    start_mock.assert_awaited_once()
    assert extension.client is replacement_client


@pytest.mark.asyncio
async def test_on_deinit_waits_for_in_flight_audio():
    client = _RaceClient()
    extension = _new_extension(client)
    stop_mock = AsyncMock()
    extension.stop_connection = stop_mock

    audio_task = asyncio.create_task(
        extension.send_audio(_frame(), "session-1")
    )
    await client.audio_send_started.wait()
    deinit_task = asyncio.create_task(extension.on_deinit(extension.ten_env))
    await asyncio.sleep(0)

    stop_mock.assert_not_awaited()
    client.allow_audio_send.set()
    await asyncio.wait_for(asyncio.gather(audio_task, deinit_task), timeout=1)

    stop_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_reentrant_restart_does_not_wait_on_connection_lock():
    extension = _new_extension(MagicMock())
    extension.stop_connection = AsyncMock()

    async def reentrant_start() -> None:
        await extension._restart_connection()

    extension.start_connection = AsyncMock(side_effect=reentrant_start)

    await asyncio.wait_for(extension._restart_connection(), timeout=1)

    extension.stop_connection.assert_awaited_once()
    extension.start_connection.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancelled_audio_send_releases_connection_lock():
    client = _RaceClient()
    extension = _new_extension(client)

    audio_task = asyncio.create_task(
        extension.send_audio(_frame(), "session-1")
    )
    await client.audio_send_started.wait()
    audio_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await audio_task

    assert extension._connection_io_lock.locked() is False
    stop_mock, start_mock = _install_reconnect_mocks(extension, client)

    await asyncio.wait_for(extension._restart_connection(), timeout=1)

    stop_mock.assert_awaited_once()
    start_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_restart_error_releases_connection_lock():
    client = MagicMock()
    client.connected = True
    extension = _new_extension(client)
    extension.stop_connection = AsyncMock(
        side_effect=RuntimeError("stop failed")
    )
    extension.start_connection = AsyncMock()

    with pytest.raises(RuntimeError, match="stop failed"):
        await asyncio.wait_for(extension._restart_connection(), timeout=1)

    assert extension._connection_io_lock.locked() is False

    stop_mock, start_mock = _install_reconnect_mocks(extension, client)
    await asyncio.wait_for(extension._restart_connection(), timeout=1)

    stop_mock.assert_awaited_once()
    start_mock.assert_awaited_once()
