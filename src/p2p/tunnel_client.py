"""Outbound Reverse Tunnel Client for matrix-compute-node (Milestone 3 NAT Traversal)."""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from typing import Any, Dict, Iterator, Optional, Tuple

import grpc
import grpc.aio

from matrix_contracts.proto import compute_service_pb2 as pb2
from matrix_contracts.proto import compute_service_pb2_grpc as pb2_grpc
from matrix_contracts.proto.compute_service_pb2 import (
    RelayTunnelMessage,
    TunnelHeartbeat,
    TunnelRegistration,
    TunnelRequest,
    TunnelResponse,
    WorkerTunnelMessage,
)
from src.crypto.signer import WorkerSigner
from src.p2p.servicer import MatrixComputeServicer

logger = logging.getLogger(__name__)


class TunnelRpcAbort(Exception):
    """Exception raised when a servicer calls context.abort()."""

    def __init__(self, code: grpc.StatusCode, details: str) -> None:
        super().__init__(f"{code}: {details}")
        self.code = code
        self.details = details


class TunnelServicerContext(grpc.ServicerContext):
    """
    Lightweight in-memory ServicerContext providing metadata and abort()
    semantics for requests arriving over the reverse tunnel.
    """

    def __init__(self, metadata: Optional[Dict[str, str]] = None) -> None:
        self._metadata = [(k.lower(), v) for k, v in (metadata or {}).items()]
        self._code = grpc.StatusCode.OK
        self._details = ""

    def invocation_metadata(self) -> list:
        return list(self._metadata)

    def abort(self, code: grpc.StatusCode, details: str) -> None:
        self._code = code
        self._details = details
        raise TunnelRpcAbort(code, details)

    def abort_with_status(self, status: Any) -> None:
        self._code = getattr(status, "code", grpc.StatusCode.INTERNAL)
        self._details = getattr(status, "details", str(status))
        raise TunnelRpcAbort(self._code, self._details)

    def set_code(self, code: grpc.StatusCode) -> None:
        self._code = code

    def set_details(self, details: str) -> None:
        self._details = details

    def is_active(self) -> bool:
        return True

    def time_remaining(self) -> Optional[float]:
        return 300.0

    def cancel(self) -> None:
        pass

    def add_callback(self, callback: Any) -> None:
        pass

    def auth_context(self) -> dict:
        return {}

    def peer(self) -> str:
        return "reverse-tunnel-peer"

    def peer_identities(self) -> tuple:
        return ()

    def peer_identity_key(self) -> Optional[str]:
        return None

    def send_initial_metadata(self, initial_metadata: Any) -> None:
        pass

    def set_trailing_metadata(self, trailing_metadata: Any) -> None:
        pass

    def trailing_metadata(self) -> list:
        return []


class ReverseTunnelClient:
    """
    Reverse Tunnel Client establishing an outbound persistent gRPC stream
    to the Discovery Relay's MatrixTunnelService.
    Traverses residential NAT and CGNAT with zero inbound worker ports.
    """

    def __init__(
        self,
        relay_endpoint: str,
        signer: WorkerSigner,
        servicer: MatrixComputeServicer,
        heartbeat_interval_seconds: float = 30.0,
        client_version: str = "matrix-compute-node/0.1.0",
    ) -> None:
        self.relay_endpoint = relay_endpoint.strip()
        self.signer = signer
        self.servicer = servicer
        self.heartbeat_interval = heartbeat_interval_seconds
        self.client_version = client_version

        self.tunnel_id: Optional[str] = None
        self.is_connected: bool = False
        self.registered_event = threading.Event()
        self.stop_event = threading.Event()

        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._send_queue: Optional[asyncio.Queue[Optional[WorkerTunnelMessage]]] = None

        # Tracking for multi-chunk streaming requests (ExecuteTileTask)
        # request_id -> (thread_queue, worker_thread_task)
        self._active_stream_queues: Dict[str, queue.Queue[Optional[pb2.TileDataChunk]]] = {}
        self._active_stream_locks: threading.Lock = threading.Lock()

    def start(self, wait_for_registration: bool = False, timeout: float = 10.0) -> bool:
        """Starts the reverse tunnel background thread."""
        if self._thread is not None and self._thread.is_alive():
            return True

        self.stop_event.clear()
        self.registered_event.clear()
        self._thread = threading.Thread(
            target=self._run_thread,
            name="ReverseTunnelClientThread",
            daemon=True,
        )
        self._thread.start()

        if wait_for_registration:
            return self.registered_event.wait(timeout=timeout)
        return True

    @property
    def is_registered(self) -> bool:
        """Returns True if the reverse tunnel session is registered and active."""
        return self.registered_event.is_set()

    async def wait_until_connected(self, timeout: float = 5.0) -> bool:
        """Asynchronously waits until the tunnel session is active and registered."""
        start_t = time.time()
        while time.time() - start_t < timeout:
            if self.is_connected and self.registered_event.is_set():
                return True
            await asyncio.sleep(0.05)
        return False

    def stop(self, timeout: float = 3.0) -> None:
        """Terminates the reverse tunnel and closes the background thread."""
        self.stop_event.set()
        if self._loop is not None and self._loop.is_running():
            if self._send_queue is not None:
                self._loop.call_soon_threadsafe(self._send_queue.put_nowait, None)
            self._loop.call_soon_threadsafe(self._cancel_all_tasks)

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            self._thread = None

        # Release any pending stream queues
        with self._active_stream_locks:
            for q in self._active_stream_queues.values():
                try:
                    q.put_nowait(None)
                except Exception:
                    pass
            self._active_stream_queues.clear()

        self.is_connected = False
        self.registered_event.clear()

    def _cancel_all_tasks(self) -> None:
        if self._loop is not None:
            for task in asyncio.all_tasks(self._loop):
                task.cancel()

    def _run_thread(self) -> None:
        """Entrypoint for the background asyncio event loop."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._tunnel_lifecycle())
        except Exception as ex:
            if not self.stop_event.is_set():
                logger.error("Reverse tunnel loop encountered error: %s", ex, exc_info=True)
        finally:
            try:
                pending = [t for t in asyncio.all_tasks(self._loop) if t is not asyncio.current_task(self._loop)]
                for task in pending:
                    task.cancel()
                if pending:
                    self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                self._loop.run_until_complete(self._loop.shutdown_asyncgens())
                self._loop.run_until_complete(asyncio.sleep(0.05))
            except Exception:
                pass
            self._loop.close()
            self._loop = None
            self.is_connected = False

    async def _tunnel_lifecycle(self) -> None:
        """Main lifecycle managing connection and reconnection loop."""
        backoff = 1.0
        while not self.stop_event.is_set():
            try:
                await self._connect_and_serve()
                backoff = 1.0
            except asyncio.CancelledError:
                break
            except Exception as ex:
                if self.stop_event.is_set():
                    break
                logger.warning("Reverse tunnel connection to %s failed: %s; retrying in %.1fs",
                               self.relay_endpoint, ex, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, 30.0)

    async def _connect_and_serve(self) -> None:
        """Establishes gRPC channel, registers tunnel, and handles duplex message flow."""
        logger.info("Opening outbound reverse tunnel to relay at %s...", self.relay_endpoint)
        channel = grpc.aio.insecure_channel(
            self.relay_endpoint,
            options=[
                ("grpc.max_receive_message_length", 512 * 1024 * 1024),
                ("grpc.max_send_message_length", 512 * 1024 * 1024),
                ("grpc.keepalive_time_ms", 30000),
                ("grpc.keepalive_timeout_ms", 10000),
                ("grpc.keepalive_permit_without_calls", 1),
            ],
        )

        try:
            stub = pb2_grpc.MatrixTunnelServiceStub(channel)
            self._send_queue = asyncio.Queue()

            async def outbound_generator():
                while not self.stop_event.is_set():
                    msg = await self._send_queue.get()
                    if msg is None:
                        break
                    yield msg

            tunnel_stream = stub.OpenTunnel(outbound_generator())

            # 1. Send TunnelRegistration
            pubkey = self.signer.get_pubkey_hex()
            now = int(time.time())
            challenge_str = f"matrix-tunnel-register:{pubkey}:{now}"
            auth_signature = self.signer.sign_message(challenge_str.encode("utf-8"))

            reg_msg = TunnelRegistration(
                worker_pubkey=pubkey,
                auth_signature=auth_signature,
                timestamp=now,
                client_version=self.client_version,
            )
            await self._send_queue.put(WorkerTunnelMessage(tunnel_id="", register=reg_msg))

            # 2. Await RegistrationAck via stream.read()
            first_msg = await tunnel_stream.read()
            if first_msg is grpc.aio.EOF or first_msg is None:
                raise ConnectionResetError("Relay closed stream before sending registration ACK")

            if not first_msg.HasField("ack"):
                raise ProtocolError(f"Expected TunnelRegistrationAck, received: {first_msg}")

            ack = first_msg.ack
            if not ack.is_accepted:
                err_msg = getattr(ack, "message", "") or getattr(ack, "error_message", "")
                raise PermissionError(f"Tunnel registration rejected by relay: {err_msg}")

            self.tunnel_id = first_msg.tunnel_id
            self.is_connected = True
            self.registered_event.set()
            logger.info("Reverse tunnel established successfully (tunnel_id=%s)", self.tunnel_id)

            # 3. Start Heartbeat background task
            heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            try:
                # 4. Incoming message dispatch loop using stream.read() consistently
                while not self.stop_event.is_set():
                    relay_msg = await tunnel_stream.read()
                    if relay_msg is grpc.aio.EOF or relay_msg is None:
                        logger.info("Relay closed reverse tunnel stream")
                        break

                    if relay_msg.HasField("heartbeat_ack"):
                        pass
                    elif relay_msg.HasField("request"):
                        # Dispatch request asynchronously
                        asyncio.create_task(self._dispatch_tunnel_request(relay_msg.request))
                    else:
                        logger.debug("Received unknown RelayTunnelMessage: %s", relay_msg)
            finally:
                heartbeat_task.cancel()
        finally:
            self.is_connected = False
            self.registered_event.clear()
            await channel.close()
            await asyncio.sleep(0.05)

    async def _heartbeat_loop(self) -> None:
        """Sends periodic TunnelHeartbeat messages to keep NAT state active."""
        while not self.stop_event.is_set():
            try:
                await asyncio.sleep(self.heartbeat_interval)
                if self.stop_event.is_set() or self._send_queue is None or not self.tunnel_id:
                    break

                hb = TunnelHeartbeat(
                    timestamp=int(time.time()),
                )
                await self._send_queue.put(
                    WorkerTunnelMessage(
                        tunnel_id=self.tunnel_id,
                        worker_pubkey=self.signer.get_pubkey_hex(),
                        heartbeat=hb,
                    )
                )
            except asyncio.CancelledError:
                break
            except Exception as ex:
                logger.warning("Error in reverse tunnel heartbeat loop: %s", ex)

    async def _dispatch_tunnel_request(self, req: TunnelRequest) -> None:
        """Routes incoming TunnelRequest to the appropriate servicer method."""
        rpc_method = req.rpc_method
        request_id = req.request_id
        task_id = req.task_id

        try:
            if rpc_method == "RequestTileTask":
                await self._handle_request_tile_task(req)
            elif rpc_method == "ExecuteTileTask":
                await self._handle_execute_tile_task_chunk(req)
            elif rpc_method == "LeaseMatrix":
                await self._handle_lease_matrix(req)
            elif rpc_method == "StepIterative":
                await self._handle_step_iterative(req)
            elif rpc_method == "ReleaseLease":
                await self._handle_release_lease(req)
            else:
                await self._send_error_response(
                    request_id=request_id,
                    task_id=task_id,
                    error_message=f"Unsupported RPC method '{rpc_method}'",
                    error_code=int(grpc.StatusCode.UNIMPLEMENTED.value[0]),
                )
        except Exception as exc:
            logger.exception("Error processing tunnel request %s (%s): %s", request_id, rpc_method, exc)
            await self._send_error_response(
                request_id=request_id,
                task_id=task_id,
                error_message=str(exc),
                error_code=int(grpc.StatusCode.INTERNAL.value[0]),
            )

    async def _handle_request_tile_task(self, req: TunnelRequest) -> None:
        request_pb = pb2.TileTaskRequest()
        request_pb.ParseFromString(req.payload_bytes)

        context = TunnelServicerContext(metadata=dict(req.metadata))
        loop = asyncio.get_running_loop()

        try:
            ack_pb = await loop.run_in_executor(None, self.servicer.RequestTileTask, request_pb, context)
            resp = TunnelResponse(
                request_id=req.request_id,
                task_id=req.task_id,
                payload_bytes=ack_pb.SerializeToString(),
                is_error=False,
            )
            if self._send_queue is not None:
                await self._send_queue.put(WorkerTunnelMessage(tunnel_id=self.tunnel_id or "", response=resp))
        except TunnelRpcAbort as abort_ex:
            await self._send_error_response(
                request_id=req.request_id,
                task_id=req.task_id,
                error_message=abort_ex.details,
                error_code=int(abort_ex.code.value[0]),
            )
        except Exception as exc:
            await self._send_error_response(
                request_id=req.request_id,
                task_id=req.task_id,
                error_message=str(exc),
                error_code=int(grpc.StatusCode.INTERNAL.value[0]),
            )

    async def _handle_execute_tile_task_chunk(self, req: TunnelRequest) -> None:
        """Handles streaming chunks for ExecuteTileTask."""
        chunk = pb2.TileDataChunk()
        chunk.ParseFromString(req.payload_bytes)

        req_id = req.request_id
        is_first_chunk = False

        with self._active_stream_locks:
            if req_id not in self._active_stream_queues:
                q: queue.Queue[Optional[pb2.TileDataChunk]] = queue.Queue()
                self._active_stream_queues[req_id] = q
                is_first_chunk = True
            else:
                q = self._active_stream_queues[req_id]

        if is_first_chunk:
            # Spawn worker thread task to consume the chunk iterator and run servicer
            loop = asyncio.get_running_loop()
            context = TunnelServicerContext(metadata=dict(req.metadata))
            asyncio.create_task(self._run_execute_stream_worker(req_id, req.task_id, q, context))

        # Deliver chunk to worker queue
        q.put(chunk)

        if req.is_final_chunk:
            q.put(None)  # Signals end of stream to chunk iterator

    async def _run_execute_stream_worker(
        self,
        request_id: str,
        task_id: str,
        q: queue.Queue[Optional[pb2.TileDataChunk]],
        context: TunnelServicerContext,
    ) -> None:
        """Worker task consuming chunk queue and invoking servicer.ExecuteTileTask."""
        def chunk_iterator() -> Iterator[pb2.TileDataChunk]:
            while True:
                item = q.get()
                if item is None:
                    break
                yield item

        loop = asyncio.get_running_loop()
        try:
            result_pb = await loop.run_in_executor(
                None,
                self.servicer.ExecuteTileTask,
                chunk_iterator(),
                context,
            )
            resp = TunnelResponse(
                request_id=request_id,
                task_id=task_id,
                payload_bytes=result_pb.SerializeToString(),
                is_error=False,
            )
            if self._send_queue is not None:
                await self._send_queue.put(WorkerTunnelMessage(tunnel_id=self.tunnel_id or "", response=resp))
        except TunnelRpcAbort as abort_ex:
            await self._send_error_response(
                request_id=request_id,
                task_id=task_id,
                error_message=abort_ex.details,
                error_code=int(abort_ex.code.value[0]),
            )
        except Exception as exc:
            logger.exception("ExecuteTileTask stream execution failed for %s: %s", task_id, exc)
            await self._send_error_response(
                request_id=request_id,
                task_id=task_id,
                error_message=str(exc),
                error_code=int(grpc.StatusCode.INTERNAL.value[0]),
            )
        finally:
            with self._active_stream_locks:
                self._active_stream_queues.pop(request_id, None)

    async def _handle_lease_matrix(self, req: TunnelRequest) -> None:
        request_pb = pb2.LeaseMatrixRequest()
        request_pb.ParseFromString(req.payload_bytes)

        context = TunnelServicerContext(metadata=dict(req.metadata))
        loop = asyncio.get_running_loop()

        try:
            ack_pb = await loop.run_in_executor(None, self.servicer.LeaseMatrix, request_pb, context)
            resp = TunnelResponse(
                request_id=req.request_id,
                task_id=req.task_id,
                payload_bytes=ack_pb.SerializeToString(),
                is_error=False,
            )
            if self._send_queue is not None:
                await self._send_queue.put(WorkerTunnelMessage(tunnel_id=self.tunnel_id or "", response=resp))
        except TunnelRpcAbort as abort_ex:
            await self._send_error_response(
                request_id=req.request_id,
                task_id=req.task_id,
                error_message=abort_ex.details,
                error_code=int(abort_ex.code.value[0]),
            )
        except Exception as exc:
            await self._send_error_response(
                request_id=req.request_id,
                task_id=req.task_id,
                error_message=str(exc),
                error_code=int(grpc.StatusCode.INTERNAL.value[0]),
            )

    async def _handle_step_iterative(self, req: TunnelRequest) -> None:
        request_pb = pb2.IterativeStepRequest()
        request_pb.ParseFromString(req.payload_bytes)

        context = TunnelServicerContext(metadata=dict(req.metadata))
        loop = asyncio.get_running_loop()

        try:
            res_pb = await loop.run_in_executor(None, self.servicer.StepIterative, request_pb, context)
            resp = TunnelResponse(
                request_id=req.request_id,
                task_id=req.task_id,
                payload_bytes=res_pb.SerializeToString(),
                is_error=False,
            )
            if self._send_queue is not None:
                await self._send_queue.put(WorkerTunnelMessage(tunnel_id=self.tunnel_id or "", response=resp))
        except TunnelRpcAbort as abort_ex:
            await self._send_error_response(
                request_id=req.request_id,
                task_id=req.task_id,
                error_message=abort_ex.details,
                error_code=int(abort_ex.code.value[0]),
            )
        except Exception as exc:
            await self._send_error_response(
                request_id=req.request_id,
                task_id=req.task_id,
                error_message=str(exc),
                error_code=int(grpc.StatusCode.INTERNAL.value[0]),
            )

    async def _handle_release_lease(self, req: TunnelRequest) -> None:
        request_pb = pb2.ReleaseLeaseRequest()
        request_pb.ParseFromString(req.payload_bytes)

        context = TunnelServicerContext(metadata=dict(req.metadata))
        loop = asyncio.get_running_loop()

        try:
            res_pb = await loop.run_in_executor(None, self.servicer.ReleaseLease, request_pb, context)
            resp = TunnelResponse(
                request_id=req.request_id,
                task_id=req.task_id,
                payload_bytes=res_pb.SerializeToString(),
                is_error=False,
            )
            if self._send_queue is not None:
                await self._send_queue.put(WorkerTunnelMessage(tunnel_id=self.tunnel_id or "", response=resp))
        except TunnelRpcAbort as abort_ex:
            await self._send_error_response(
                request_id=req.request_id,
                task_id=req.task_id,
                error_message=abort_ex.details,
                error_code=int(abort_ex.code.value[0]),
            )
        except Exception as exc:
            await self._send_error_response(
                request_id=req.request_id,
                task_id=req.task_id,
                error_message=str(exc),
                error_code=int(grpc.StatusCode.INTERNAL.value[0]),
            )

    async def _send_error_response(
        self,
        request_id: str,
        task_id: str,
        error_message: str,
        error_code: int = 13,
    ) -> None:
        resp = TunnelResponse(
            request_id=request_id,
            task_id=task_id,
            payload_bytes=b"",
            is_error=True,
            error_message=error_message,
            grpc_status_code=error_code,
        )
        if self._send_queue is not None and not self.stop_event.is_set():
            await self._send_queue.put(WorkerTunnelMessage(tunnel_id=self.tunnel_id or "", response=resp))
