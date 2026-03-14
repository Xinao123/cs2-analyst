import unittest

from alerts.telegram import _format_short_datetime


class TelegramDateTimeFormatTests(unittest.TestCase):
    def test_format_short_datetime_with_brt_label(self):
        rendered = _format_short_datetime(
            "2026-03-14T18:00:00Z",
            tz_name="America/Sao_Paulo",
            tz_label="BRT (Brasília)",
        )
        self.assertEqual("14/03 15:00 BRT (Brasília)", rendered)


if __name__ == "__main__":
    unittest.main()

