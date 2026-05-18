from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class MonthWindow:
    month: str
    start: datetime
    end: datetime


def parse_month(month: str) -> MonthWindow:
    start = datetime.strptime(month, "%Y-%m")
    end = datetime(start.year + (start.month == 12), start.month % 12 + 1, 1)
    return MonthWindow(month=month, start=start, end=end)


def month_minus(month: str, n: int) -> str:
    year, mon = map(int, month.split("-"))
    mon -= n
    while mon <= 0:
        year -= 1
        mon += 12
    return f"{year:04d}-{mon:02d}"

