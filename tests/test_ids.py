from finpipe.ids import description_key, transaction_id


def test_description_key_collapses_whitespace_and_uppercases():
    assert description_key("  Blue   bottle\tCoffee \n") == "BLUE BOTTLE COFFEE"


def test_id_is_deterministic():
    a = transaction_id("checking", "2025-01-02", -4000, "ONLINE TRANSFER", 1)
    b = transaction_id("checking", "2025-01-02", -4000, "ONLINE TRANSFER", 1)
    assert a == b
    assert len(a) == 64


def test_every_identity_field_changes_the_id():
    base = ("checking", "2025-01-02", -4000, "ONLINE TRANSFER", 1)
    ids = {transaction_id(*base)}
    ids.add(transaction_id("credit_card", *base[1:]))
    ids.add(transaction_id(base[0], "2025-01-03", *base[2:]))
    ids.add(transaction_id(base[0], base[1], -4001, *base[3:]))
    ids.add(transaction_id(*base[:3], "ONLINE TRANSFER 2", base[4]))
    ids.add(transaction_id(*base[:4], 2))
    assert len(ids) == 6


def test_digits_cannot_migrate_between_amount_and_description():
    # Both payloads would read "...100AB1" without a separator between fields.
    one = transaction_id("checking", "2025-01-02", 100, "AB", 1)
    two = transaction_id("checking", "2025-01-02", 10, "0AB", 1)
    assert one != two
