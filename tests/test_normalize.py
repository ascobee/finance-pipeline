import pytest

from finpipe.normalize import normalize_merchant


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("SQ *BLUE BOTTLE COFFEE 0423 DENVER CO", "Blue Bottle Coffee"),
        ("BLUE BOTTLE COFFEE BOULDER", "Blue Bottle Coffee"),
        ("AMZN Mktp US*2K4LM8XY", "Amazon"),
        ("AMAZON.COM*2XB9WM9P AMZN.COM/BILL WA", "Amazon"),
        ("WHOLEFDS DEN 14188", "Whole Foods"),
        ("POS DEBIT KING SOOPERS #7525 AURORA CO", "King Soopers"),
        ("UBER   *TRIP HELP.UBER.COM CA", "Uber"),
        ("DD *DOORDASH AB12CD", "DoorDash"),
        ("TST* THE PIZZA PLACE DENVER CO", "The Pizza Place"),
        ("PAYROLL ACME CORP DIRECT DEP 433218", "Payroll Acme Corp Direct Dep"),
        ("XCEL ENERGY AUTOPAY 118203", "Xcel Energy Autopay"),
        ("CREDIT CARD PAYMENT - THANK YOU", "Credit Card Payment Thank You"),
        ("ONLINE TRANSFER TO SAVINGS 6184", "Online Transfer To Savings"),
    ],
)
def test_known_descriptions(raw, expected):
    assert normalize_merchant(raw) == expected


def test_store_numbers_and_cities_do_not_split_a_merchant():
    variants = [
        "SQ *BLUE BOTTLE COFFEE 0423 DENVER CO",
        "SQ *BLUE BOTTLE COFFEE 9981 BOULDER CO",
        "BLUE BOTTLE COFFEE ARVADA",
    ]
    assert len({normalize_merchant(v) for v in variants}) == 1


def test_case_and_whitespace_are_ignored():
    assert normalize_merchant("  starbucks   store 123 ") == normalize_merchant("STARBUCKS STORE 123")


@pytest.mark.parametrize("raw", [None, "", "   ", "12345", "#99"])
def test_unusable_input_becomes_unknown(raw):
    assert normalize_merchant(raw) == "Unknown"
