"""
Time helpers to keep a single UTC storage policy and local-time display policy.
"""

from __future__ import annotations

import logging
from datetime import datetime, time, timezone
from typing import Any
from zoneinfo import ZoneInfo

DEFAULT_DISPLAY_TZ = "America/Sao_Paulo"

_DATETIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d",
    "%d-%m-%Y %H:%M:%S",
    "%d-%m-%Y",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y",
)

_DATE_ONLY_FORMATS = ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y")
_TIME_ONLY_FORMATS = ("%H:%M", "%H:%M:%S")


def resolve_timezone(tz_name: str | None, logger: logging.Logger | None = None) -> ZoneInfo:
    name = str(tz_name or DEFAULT_DISPLAY_TZ).strip() or DEFAULT_DISPLAY_TZ
    try:
        return ZoneInfo(name)
    except Exception:
        if logger:
            logger.warning("[TIME] Timezone invalida (%s), usando UTC.", name)
        return ZoneInfo("UTC")


def parse_datetime_to_utc(
    value: Any,
    assume_tz: str | ZoneInfo | None = None,
    logger: logging.Logger | None = None,
    context: str = "",
) -> datetime | None:
    """
    Parse datetime-like values and normalize to aware UTC datetime.

    If input is naive and `assume_tz` is absent, fallback is UTC for legacy
    compatibility and an optional DEBUG log is emitted.
    """
    if value is None:
        return None

    dt = None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            for fmt in _DATETIME_FORMATS:
                try:
                    dt = datetime.strptime(text, fmt)
                    break
                except ValueError:
                    continue
            if dt is None:
                return None

    if dt.tzinfo is None:
        if assume_tz:
            tz = assume_tz if isinstance(assume_tz, ZoneInfo) else resolve_timezone(str(assume_tz), logger=logger)
            dt = dt.replace(tzinfo=tz)
        else:
            if logger:
                where = f" ({context})" if context else ""
                logger.debug("[TIME] Datetime naive tratado como UTC%s: %s", where, _safe_text(value))
            dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc)


def parse_date_time_to_utc(
    date_value: Any,
    time_value: Any = None,
    assume_tz: str | ZoneInfo | None = None,
    logger: logging.Logger | None = None,
    context: str = "",
) -> datetime | None:
    """
    Parse date + optional time and normalize to aware UTC datetime.
    """
    if date_value is None:
        return None

    date_text = str(date_value).strip()
    if not date_text:
        return None

    # Fast path: date already contains full datetime (possibly with tz).
    direct = parse_datetime_to_utc(
        date_text,
        assume_tz=assume_tz,
        logger=logger,
        context=context,
    )
    if direct and not _is_date_only_string(date_text):
        if time_value and direct.hour == 0 and direct.minute == 0:
            parsed_time = _parse_time_only(time_value)
            if parsed_time:
                base_tz = (
                    resolve_timezone(str(assume_tz), logger=logger)
                    if assume_tz
                    else timezone.utc
                )
                local_dt = direct.astimezone(base_tz).replace(
                    hour=parsed_time.hour,
                    minute=parsed_time.minute,
                    second=0,
                    microsecond=0,
                )
                return local_dt.astimezone(timezone.utc)
        return direct

    # Date-only fallback with optional time token.
    parsed_date = _parse_date_only(date_text)
    if not parsed_date:
        return direct

    parsed_time = _parse_time_only(time_value) or time(0, 0, 0)
    naive = datetime(
        year=parsed_date.year,
        month=parsed_date.month,
        day=parsed_date.day,
        hour=parsed_time.hour,
        minute=parsed_time.minute,
        second=parsed_time.second,
    )

    if assume_tz:
        tz = assume_tz if isinstance(assume_tz, ZoneInfo) else resolve_timezone(str(assume_tz), logger=logger)
        aware = naive.replace(tzinfo=tz)
    else:
        if logger:
            where = f" ({context})" if context else ""
            logger.debug("[TIME] Data/hora naive tratada como UTC%s: %s %s", where, date_text, _safe_text(time_value))
        aware = naive.replace(tzinfo=timezone.utc)

    return aware.astimezone(timezone.utc)


def to_storage_utc_iso(
    value: Any,
    assume_tz: str | ZoneInfo | None = None,
    logger: logging.Logger | None = None,
    context: str = "",
) -> str:
    """
    Serialize to UTC naive ISO string for DB storage compatibility.
    """
    dt_utc = parse_datetime_to_utc(
        value,
        assume_tz=assume_tz,
        logger=logger,
        context=context,
    )
    if not dt_utc:
        return _safe_text(value)
    return dt_utc.replace(tzinfo=None).isoformat(timespec="seconds")


def to_storage_utc_datetime(
    value: Any,
    assume_tz: str | ZoneInfo | None = None,
    logger: logging.Logger | None = None,
    context: str = "",
) -> datetime | None:
    dt_utc = parse_datetime_to_utc(
        value,
        assume_tz=assume_tz,
        logger=logger,
        context=context,
    )
    if not dt_utc:
        return None
    return dt_utc.replace(tzinfo=None)


def format_datetime_for_timezone(
    value: Any,
    tz_name: str = DEFAULT_DISPLAY_TZ,
    fmt: str = "%d/%m %H:%M",
    tz_suffix: str = "",
    logger: logging.Logger | None = None,
) -> str:
    dt_utc = parse_datetime_to_utc(value, logger=logger, context="format_datetime_for_timezone")
    if not dt_utc:
        return _safe_text(value) or "Data indefinida"
    tz = resolve_timezone(tz_name, logger=logger)
    local = dt_utc.astimezone(tz)
    base = local.strftime(fmt)
    suffix = f" {tz_suffix.strip()}" if tz_suffix and tz_suffix.strip() else ""
    return f"{base}{suffix}"


def format_date_for_timezone(
    value: Any,
    tz_name: str = DEFAULT_DISPLAY_TZ,
    fmt: str = "%d/%m/%Y",
    logger: logging.Logger | None = None,
) -> str:
    text = _safe_text(value)
    if text:
        date_only = _parse_date_only(text)
        if date_only is not None:
            return date_only.strftime(fmt)

    dt_utc = parse_datetime_to_utc(value, logger=logger, context="format_date_for_timezone")
    if not dt_utc:
        return _safe_text(value) or "Data indefinida"
    tz = resolve_timezone(tz_name, logger=logger)
    return dt_utc.astimezone(tz).strftime(fmt)


def _parse_date_only(value: str) -> datetime | None:
    for fmt in _DATE_ONLY_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _parse_time_only(value: Any) -> time | None:
    text = _safe_text(value)
    if not text:
        return None
    for fmt in _TIME_ONLY_FORMATS:
        try:
            return datetime.strptime(text, fmt).time()
        except ValueError:
            continue
    return None


def _is_date_only_string(value: str) -> bool:
    return _parse_date_only(value) is not None


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()
