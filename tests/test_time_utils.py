import unittest
from datetime import timezone

from utils.time_utils import (
    format_datetime_for_timezone,
    parse_date_time_to_utc,
    parse_datetime_to_utc,
    to_storage_utc_iso,
)


class TimeUtilsTests(unittest.TestCase):
    def test_parse_iso_offset_to_utc(self):
        parsed = parse_datetime_to_utc("2026-03-14T18:00:00+01:00")
        self.assertIsNotNone(parsed)
        self.assertEqual(timezone.utc, parsed.tzinfo)
        self.assertEqual(17, parsed.hour)

    def test_parse_hltv_naive_with_assume_tz(self):
        parsed = parse_date_time_to_utc(
            "14-03-2026",
            "18:00",
            assume_tz="America/Sao_Paulo",
        )
        self.assertIsNotNone(parsed)
        self.assertEqual(timezone.utc, parsed.tzinfo)
        self.assertEqual(21, parsed.hour)

    def test_storage_iso_is_utc_naive(self):
        value = to_storage_utc_iso("2026-03-14T18:00:00Z")
        self.assertEqual("2026-03-14T18:00:00", value)

    def test_display_format_converts_to_brt_with_label(self):
        rendered = format_datetime_for_timezone(
            "2026-03-14T18:00:00Z",
            tz_name="America/Sao_Paulo",
            fmt="%d/%m %H:%M",
            tz_suffix="BRT (Brasília)",
        )
        self.assertEqual("14/03 15:00 BRT (Brasília)", rendered)


if __name__ == "__main__":
    unittest.main()

