"""Bounded downloads of user files; no arbitrary URL or local-path MCP access."""

import asyncio
import ipaddress
import os
import socket
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from server.core.errors import ModelError
from server.core.limits import MAX_FILE_BYTES, TRANSFER_TIMEOUT_SECONDS
from server.schemas.files import UploadedFile

DOWNLOAD_TIMEOUT_SECONDS = TRANSFER_TIMEOUT_SECONDS


def model_filename(file: UploadedFile, filename: str | None = None) -> str:
    name = file.file_name or filename or ""
    if not name:
        raise ModelError("missing_filename", "Provide filename ending in .lp or .mps.")
    name = name.replace("\\", "/").rsplit("/", 1)[-1]
    if len(name) > 512 or "\x00" in name or Path(name).suffix.lower() not in {".lp", ".mps"}:
        raise ModelError("unsupported_format", "Only .lp and .mps model files are supported.")
    return name


async def public_download_address(url: str, allowed_hosts: set[str]) -> tuple[str, str]:
    """Resolve once, reject private addresses, and pin the HTTP connection to that IP."""
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
        valid = (
            parts.scheme == "https"
            and parts.port in (None, 443)
            and not parts.username
            and not parts.password
            and not parts.fragment
            and host in allowed_hosts
        )
    except ValueError:
        valid = False
    if not valid:
        raise ModelError("download_url_not_allowed", "File URL must use an allowed HTTPS host.")
    try:
        addresses = await asyncio.get_running_loop().getaddrinfo(host, 443, type=socket.SOCK_STREAM)
        ips = [ipaddress.ip_address(address[4][0]) for address in addresses]
    except (OSError, ValueError) as exc:
        raise ModelError("download_failed", "Could not resolve the file host.") from exc
    if not ips or any(not ip.is_global or ip.is_multicast for ip in ips):
        raise ModelError("download_url_not_allowed", "File host must resolve to public addresses.")
    # httpcore's SNI override preserves TLS hostname verification while connecting
    # to the previously checked IP, preventing a second DNS lookup/rebinding.
    pinned_url = str(httpx.URL(url).copy_with(host=str(ips[0])))
    return pinned_url, host


async def download_file(file: UploadedFile, destination: Path) -> None:
    allowed_hosts = {
        host.strip().lower()
        for host in os.environ.get("OR_LENS_DOWNLOAD_HOSTS", "").split(",")
        if host.strip()
    }
    if not allowed_hosts:
        raise ModelError(
            "download_not_configured",
            "Set OR_LENS_DOWNLOAD_HOSTS to trusted upload-storage hostnames on the server.",
        )
    try:
        async with asyncio.timeout(DOWNLOAD_TIMEOUT_SECONDS):
            url, host = await public_download_address(file.download_url, allowed_hosts)
            async with httpx.AsyncClient(
                follow_redirects=False, trust_env=False, timeout=DOWNLOAD_TIMEOUT_SECONDS
            ) as client:
                async with client.stream(
                    "GET",
                    url,
                    headers={"Host": host, "Accept-Encoding": "identity"},
                    extensions={"sni_hostname": host},
                ) as response:
                    if response.status_code != 200:
                        raise ModelError("download_failed", "File download failed or expired.")
                    if response.headers.get("content-encoding", "identity").lower() != "identity":
                        raise ModelError(
                            "download_failed", "Compressed file responses are unsupported."
                        )
                    length = response.headers.get("content-length")
                    if length and (not length.isdecimal() or int(length) > MAX_FILE_BYTES):
                        raise ModelError("file_too_large", "Model file exceeds 1 GiB.")
                    size = 0
                    with destination.open("wb") as stream:
                        async for chunk in response.aiter_bytes(chunk_size=64 * 1024):
                            size += len(chunk)
                            if size > MAX_FILE_BYTES:
                                raise ModelError("file_too_large", "Model file exceeds 1 GiB.")
                            stream.write(chunk)
                    if size == 0:
                        raise ModelError("empty_file", "Model file is empty.")
    except TimeoutError as exc:
        raise ModelError("download_timeout", "File download exceeded its time limit.") from exc
    except (httpx.HTTPError, OSError) as exc:
        raise ModelError("download_failed", "Could not download the model file.") from exc
