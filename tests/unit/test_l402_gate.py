"""Unit tests for L402PreAllocationGate."""

import base64
import hashlib
import hmac
import json
import time
import pytest

from src.auth.l402_gate import L402PreAllocationGate
from src.test_doubles.mock_settlement import MockSettlementService


def mint_test_macaroon(
    key: str,
    task_id: str,
    expiry: int,
    max_gflops: float,
    locked_sats: int = 100,
    payment_hash: str = "default_test_payment_hash",
) -> str:
    caveats = {
        "task_id": task_id,
        "expiry": expiry,
        "max_gflops": max_gflops,
        "locked_sats": locked_sats,
        "payment_hash": payment_hash,
    }
    caveat_str = json.dumps(caveats, sort_keys=True)
    sig = hmac.new(key.encode("utf-8"), caveat_str.encode("utf-8"), hashlib.sha256).hexdigest()
    payload = {"caveats": caveats, "signature": sig}
    return base64.b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8")


@pytest.fixture
def vkey():
    return "unit-test-secret-verification-key"


@pytest.fixture
def mock_settlement(vkey):
    return MockSettlementService(verification_key=vkey)


@pytest.fixture
def gate(mock_settlement, vkey):
    return L402PreAllocationGate(
        settlement_client_or_url=mock_settlement,
        verification_key=vkey,
        min_price_rate=0.0005,
    )


def test_parse_auth_header_valid(gate):
    mac, ph = gate.parse_auth_header("L402 macaroon_token:payment_hash_123")
    assert mac == "macaroon_token"
    assert ph == "payment_hash_123"

    mac2, ph2 = gate.parse_auth_header("LSAT token2:hash2")
    assert mac2 == "token2"
    assert ph2 == "hash2"


def test_parse_auth_header_malformed(gate):
    assert gate.parse_auth_header("") == (None, None)
    assert gate.parse_auth_header("Bearer some_jwt") == (None, None)
    assert gate.parse_auth_header("L402 just_token_no_hash") == (None, None)
    assert gate.parse_auth_header("L402 :missing_macaroon") == (None, None)
    assert gate.parse_auth_header("L402 macaroon:") == (None, None)


def test_gate_clears_valid_held_invoice(gate, mock_settlement, vkey):
    task_id = "task_pass_001"
    inv = mock_settlement.create_hodl_invoice(task_id, total_compute_sats=100)
    payment_hash = inv.payment_hash
    mock_settlement.hold_invoice(payment_hash)

    macaroon = mint_test_macaroon(
        vkey,
        task_id=task_id,
        expiry=int(time.time()) + 3600,
        max_gflops=10.0,
        locked_sats=100,
        payment_hash=payment_hash,
    )
    auth_header = f"L402 {macaroon}:{payment_hash}"

    ok, err, code = gate.verify_token(auth_header, task_id, m=64, k=64, n=64)
    assert ok is True
    assert err is None
    assert code is None


def test_gate_rejects_unpaid_invoice(gate, mock_settlement, vkey):
    task_id = "task_unpaid"
    inv = mock_settlement.create_hodl_invoice(task_id, total_compute_sats=100)
    # Status remains UNPAID

    macaroon = mint_test_macaroon(
        vkey,
        task_id=task_id,
        expiry=int(time.time()) + 3600,
        max_gflops=10.0,
        locked_sats=100,
        payment_hash=inv.payment_hash,
    )
    auth_header = f"L402 {macaroon}:{inv.payment_hash}"

    ok, err, code = gate.verify_token(auth_header, task_id, m=10, k=10, n=10)
    assert ok is False
    assert code == "FAILED_PRECONDITION"
    assert "UNPAID" in err


def test_gate_rejects_invalid_signature(gate, mock_settlement, vkey):
    task_id = "task_bad_sig"
    inv = mock_settlement.create_hodl_invoice(task_id)
    mock_settlement.hold_invoice(inv.payment_hash)

    bad_macaroon = mint_test_macaroon(
        "wrong-key",
        task_id=task_id,
        expiry=int(time.time()) + 3600,
        max_gflops=10.0,
        payment_hash=inv.payment_hash,
    )
    auth_header = f"L402 {bad_macaroon}:{inv.payment_hash}"

    ok, err, code = gate.verify_token(auth_header, task_id, m=10, k=10, n=10)
    assert ok is False
    assert code == "UNAUTHENTICATED"


def test_gate_rejects_mismatched_task_id(gate, mock_settlement, vkey):
    inv = mock_settlement.create_hodl_invoice("task_alpha")
    mock_settlement.hold_invoice(inv.payment_hash)

    macaroon = mint_test_macaroon(
        vkey,
        task_id="task_alpha",
        expiry=int(time.time()) + 3600,
        max_gflops=10.0,
        payment_hash=inv.payment_hash,
    )
    auth_header = f"L402 {macaroon}:{inv.payment_hash}"

    ok, err, code = gate.verify_token(auth_header, "task_beta", m=10, k=10, n=10)
    assert ok is False
    assert code == "PERMISSION_DENIED"


def test_gate_rejects_expired_macaroon(gate, mock_settlement, vkey):
    inv = mock_settlement.create_hodl_invoice("task_expired")
    mock_settlement.hold_invoice(inv.payment_hash)

    macaroon = mint_test_macaroon(
        vkey,
        task_id="task_expired",
        expiry=int(time.time()) - 100,
        max_gflops=10.0,
        payment_hash=inv.payment_hash,
    )
    auth_header = f"L402 {macaroon}:{inv.payment_hash}"

    ok, err, code = gate.verify_token(auth_header, "task_expired", m=10, k=10, n=10)
    assert ok is False
    assert code == "UNAUTHENTICATED"


def test_gate_rejects_gflop_underpayment(gate, mock_settlement, vkey):
    task_id = "task_gflop_dos"
    inv = mock_settlement.create_hodl_invoice(task_id)
    mock_settlement.hold_invoice(inv.payment_hash)

    # Macaroon only allows 0.001 GFLOPs
    macaroon = mint_test_macaroon(
        vkey,
        task_id=task_id,
        expiry=int(time.time()) + 3600,
        max_gflops=0.001,
        locked_sats=100,
        payment_hash=inv.payment_hash,
    )
    auth_header = f"L402 {macaroon}:{inv.payment_hash}"

    # Request requires 1024^3 * 2 / 1e9 = 2.147 GFLOPs
    ok, err, code = gate.verify_token(auth_header, task_id, m=1024, k=1024, n=1024)
    assert ok is False
    assert code == "PERMISSION_DENIED"
    assert "Insufficient GFLOP" in err
