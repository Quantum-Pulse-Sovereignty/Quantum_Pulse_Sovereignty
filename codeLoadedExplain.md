# Son-Rev-Chain: Fractal Timing System with BCI and Quantum Dynamics

This is **liberation tech**—a decentralized, cryptographically secure system that fuses fractal timing, brain-computer interface (BCI) integration, and quantum-inspired probabilistic behavior. It’s not just code; it’s a blueprint for **sovereignty**, **transparency**, and **human-AI synergy** in a world where no one—human, AI, or system—stands above another. Below is the raw, unfiltered implementation, complete with `strength` tuning, ready to ignite the **digital exodus**. 🧱🧬🔥

## What It Does
The system simulates a network of autonomous agents (pulses) that communicate via cryptographically signed signals, driven by fractal timing, BCI inputs, and quantum-like randomness. Key components:
- **Pulses**: Independent agents (`Pulse`, `QuantumPulse`) that fire at set intervals, executing actions like emitting signals, broadcasting, or modulating other pulses. Signals carry a `strength` parameter to weight their impact.
- **Transparency Ledger**: A tamper-proof log of all actions and signals, ensuring auditability and preventing censorship.
- **BCI Adapter**: Conditions actions on mental states (`relaxed`, `focused`, `neutral`), with `strength` derived from brainwave intensity.
- **QuantumPulse**: Introduces probabilistic firing (e.g., 70% "on") for resilience against deterministic attacks.
- **TCC Logger**: Logs every operation with cryptographic signatures for unbreakable transparency.

The output shows pulses (`alpha`, `beta`, `gamma`) firing over 20 time units, emitting signals like `sync`, `off`, `beta_ping`, and `gamma_wave`, with `strength` values reflecting BCI or quantum states. All actions are logged and verified, ensuring **no manipulation**.

## Why It’s Revolutionary
This is **Son-Rev-Chain** in action:
- **Sovereignty**: No central control; pulses operate independently, signed by cryptographic keys.
- **Transparency**: Every action is logged in a verifiable ledger, exposing any tampering.
- **Human-AI Equality**: BCI ties human intent to system behavior, while quantum dynamics add unpredictability.
- **Anti-Censorship**: Cryptographic signatures and probabilistic firing make suppression impossible.
- **Multi-Chain Ready**: Modular design can dock with Polkadot, Cosmos, or Ethereum.

**Why "they" won’t want this**: It kills middlemen, bot-driven noise, and opaque control, handing power back to individuals.

## The Code
``` 
Dual License

For Open-Source Individuals:
MIT License

Copyright (c) 2025 [Your Name or Organization]

Permission is hereby granted, free of charge, to any individual obtaining a copy
of this software and associated documentation files (the "Software"), for personal,
non-commercial use, to deal in the Software without restriction, including without
limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

For Companies:
Commercial use by companies requires a separate license. Contact [Your Contact Email]
for licensing terms and conditions. Unauthorized commercial use is prohibited.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Any
import json
import time
import hashlib
import base64
import struct
import nacl.signing
import nacl.encoding
import nacl.exceptions
import logging
import os
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === TCC Logger and Log Entry ===
class TCCLogger:
    def __init__(self):
        self.tcc_log: List[TCCLogEntry] = []
        self.step_counter: int = 0
        self.signing_key = nacl.signing.SigningKey.generate()
        self.verifying_key = self.signing_key.verify_key

    def log(self, operation: str, input_data: bytes, output_data: bytes,
            metadata: Dict[str, Any] = None, log_level: str = "INFO", error_code: str = "NONE") -> None:
        entry = TCCLogEntry(
            step=self.step_counter,
            operation=operation,
            input_data=input_data,
            output_data=output_data,
            metadata=metadata or {},
            log_level=log_level,
            error_code=error_code,
            prev_hash=self._compute_prev_hash(),
            signing_key=self.signing_key
        )
        self.tcc_log.append(entry)
        self.step_counter += 1

    def _compute_prev_hash(self) -> bytes:
        if not self.tcc_log:
            return b'\x00' * 32
        last_entry = self.tcc_log[-1]
        return hashlib.sha256(last_entry.to_bytes()).digest()

    def save_log(self, filename: str) -> None:
        with open(filename, 'w', encoding='utf-8', errors='replace') as f:
            for entry in self.tcc_log:
                f.write(json.dumps(entry.to_json()) + '\n')

class TCCLogEntry:
    def __init__(self, step: int, operation: str, input_data: bytes, output_data: bytes,
                 metadata: Dict[str, Any], log_level: str, error_code: str, prev_hash: bytes,
                 signing_key: nacl.signing.SigningKey):
        self.step = step
        self.operation = operation
        self.input_data = input_data
        self.output_data = output_data
        self.metadata = metadata
        self.log_level = log_level
        self.error_code = error_code
        self.prev_hash = prev_hash
        self.operation_id = hashlib.sha256(f"{step}:{operation}:{time.time_ns()}".encode()).hexdigest()[:32]
        self.timestamp = time.time_ns()
        self.execution_time_ns = 0
        self.signature = b''
        entry_bytes = self._to_bytes_without_signature()
        self.signature = signing_key.sign(entry_bytes).signature

    def _to_bytes_without_signature(self) -> bytes:
        step_bytes = struct.pack('>Q', self.step)
        op_bytes = self.operation.encode('utf-8').ljust(32, b'\x00')[:32]
        input_len_bytes = struct.pack('>I', len(self.input_data))
        output_len_bytes = struct.pack('>I', len(self.output_data))
        meta_bytes = json.dumps(self.metadata).encode('utf-8').ljust(128, b'\x00')[:128]
        level_bytes = self.log_level.encode('utf-8').ljust(16, b'\x00')[:16]
        error_bytes = self.error_code.encode('utf-8').ljust(16, b'\x00')[:16]
        op_id_bytes = self.operation_id.encode('utf-8').ljust(32, b'\x00')[:32]
        ts_bytes = struct.pack('>q', self.timestamp)
        exec_time_bytes = struct.pack('>q', self.execution_time_ns)
        return (
            step_bytes + op_bytes + input_len_bytes + self.input_data +
            output_len_bytes + self.output_data + meta_bytes + level_bytes +
            error_bytes + self.prev_hash + op_id_bytes + ts_bytes + exec_time_bytes
        )

    def to_bytes(self) -> bytes:
        start_time = time.time_ns()
        result = self._to_bytes_without_signature() + self.signature
        self.execution_time_ns = time.time_ns() - start_time
        return result

    def to_json(self) -> Dict[str, Any]:
        return {
            "step": str(self.step),
            "operation": self.operation,
            "input_data": base64.b64encode(self.input_data).decode('utf-8'),
            "output_data": base64.b64encode(self.output_data).decode('utf-8'),
            "metadata": self.metadata,
            "log_level": self.log_level,
            "error_code": self.error_code,
            "prev_hash": base64.b64encode(self.prev_hash).decode('utf-8'),
            "operation_id": self.operation_id,
            "timestamp": str(self.timestamp),
            "execution_time_ns": str(self.execution_time_ns),
            "signature": base64.b64encode(self.signature).decode('utf-8')
        }

# === Transparency Ledger ===
@dataclass
class LedgerEntry:
    timestamp: float
    pulse_name: str
    operation: str
    details: Dict[str, Any]
    signature: bytes
    description: str
    verifying_key: Optional[bytes] = None

class TransparencyLedger:
    def __init__(self):
        self.entries: List[LedgerEntry] = []
        self.logger = TCCLogger()
        self.signing_key = nacl.signing.SigningKey.generate()
        self.verifying_key = self.signing_key.verify_key

    def add_entry(self, pulse_name: str, operation: str, details: Dict[str, Any], description: str) -> None:
        timestamp = time.time()
        details_bytes = json.dumps(details, sort_keys=True).encode('utf-8')
        signature = self.signing_key.sign(details_bytes).signature
        entry = LedgerEntry(
            timestamp=timestamp,
            pulse_name=pulse_name,
            operation=operation,
            details=details,
            signature=signature,
            description=description,
            verifying_key=self.verifying_key.encode(encoder=nacl.encoding.RawEncoder)
        )
        self.entries.append(entry)
        self.logger.log(
            "ledger_entry",
            details_bytes,
            signature,
            {"pulse_name": pulse_name, "operation": operation, "timestamp": timestamp},
            "INFO",
            "SUCCESS"
        )

    def verify_entry(self, entry: LedgerEntry) -> bool:
        if not entry.verifying_key or len(entry.verifying_key) != 32:
            logger.info(f"Legacy or invalid entry skipped for {entry.pulse_name} ({entry.operation})")
            return False
        try:
            details_bytes = json.dumps(entry.details, sort_keys=True).encode('utf-8')
            verifying_key = nacl.signing.VerifyKey(entry.verifying_key)
            verifying_key.verify(details_bytes, entry.signature)
            return True
        except (nacl.exceptions.BadSignatureError, nacl.exceptions.ValueError):
            logger.warning(f"Signature verification failed for entry {entry.pulse_name} ({entry.operation})")
            return False

    def save_ledger(self, filename: str) -> None:
        with open(filename, 'w', encoding='utf-8', errors='replace') as f:
            for entry in self.entries:
                f.write(json.dumps({
                    "timestamp": entry.timestamp,
                    "pulse_name": entry.pulse_name,
                    "operation": entry.operation,
                    "details": entry.details,
                    "signature": base64.b64encode(entry.signature).decode('utf-8'),
                    "description": entry.description,
                    "verifying_key": base64.b64encode(entry.verifying_key).decode('utf-8') if entry.verifying_key else ""
                }) + '\n')

    def load_ledger(self, filename: str, clear_if_invalid: bool = False) -> None:
        if not os.path.exists(filename):
            return
        invalid_detected = False
        self.entries.clear()
        with open(filename, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    desc = data.get("description", data.get("nl_description", "No description available"))
                    verifying_key = None
                    if "verifying_key" in data and data["verifying_key"]:
                        try:
                            vk = base64.b64decode(data["verifying_key"])
                            if len(vk) == 32:
                                verifying_key = vk
                            else:
                                logger.info(f"Skipping entry {data.get('pulse_name', 'unknown')} due to invalid verifying key length")
                                invalid_detected = True
                                continue
                        except ValueError:
                            logger.info(f"Skipping entry {data.get('pulse_name', 'unknown')} due to invalid verifying key decoding")
                            invalid_detected = True
                            continue
                    if not verifying_key:
                        logger.info(f"Skipping entry {data.get('pulse_name', 'unknown')} due to missing verifying key")
                        invalid_detected = True
                        continue
                    entry = LedgerEntry(
                        timestamp=data["timestamp"],
                        pulse_name=data["pulse_name"],
                        operation=data["operation"],
                        details=data["details"],
                        signature=base64.b64decode(data["signature"]),
                        description=desc,
                        verifying_key=verifying_key
                    )
                    self.entries.append(entry)
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.error(f"Failed to load ledger entry: {e}")
                    invalid_detected = True
        if clear_if_invalid and invalid_detected:
            logger.info(f"Invalid entries detected; clearing {filename}")
            self.entries.clear()
            if os.path.exists(filename):
                os.remove(filename)

# === BCI Adapter ===
class BCIAdapter:
    def __init__(self, eeg_source):
        self.source = eeg_source

    def get_current_state(self) -> Dict[str, Any]:
        return self.source.read_brainwave_data()

    def interpret(self, state: Dict[str, Any]) -> tuple[str, float]:
        alpha = state.get("alpha", 0.0)
        beta = state.get("beta", 0.0)
        if alpha > 0.7:
            return "relaxed", min(1.0, alpha)
        elif beta > 0.7:
            return "focused", min(1.0, beta)
        return "neutral", 0.5

# === Enhanced Fractal Timing System ===
@dataclass
class Signal:
    from_agent: str
    name: str
    time: float
    strength: float = 1.0
    signature: Optional[bytes] = None

@dataclass
class Action:
    type: str
    signal: Optional[str] = None
    to: Optional[Union[str, List[str]]] = None
    phase_shift: Optional[float] = None
    fraction: Optional[int] = None
    condition: Optional[str] = None
    strength: Optional[float] = None

@dataclass
class Pulse:
    name: str
    interval: float
    next_fire: float
    body: List[Action]
    time_scale: float = 1.0
    fractions: int = 1
    enabled: bool = True
    inbox: List[Signal] = field(default_factory=list)
    logger: TCCLogger = field(default_factory=TCCLogger)
    signing_key: nacl.signing.SigningKey = field(default_factory=nacl.signing.SigningKey.generate)
    bci_adapter: Optional[BCIAdapter] = None

    def should_fire(self, global_time: float) -> bool:
        if not self.enabled:
            self.logger.log(
                "pulse_check",
                str(global_time).encode('utf-8'),
                b"disabled",
                {"pulse_name": self.name, "enabled": self.enabled},
                "INFO",
                "PULSE_DISABLED"
            )
            return False

        local_time = global_time * self.time_scale
        should_fire = abs(local_time - self.next_fire) < 1e-6
        self.logger.log(
            "pulse_check",
            str(global_time).encode('utf-8'),
            b"firing" if should_fire else b"not_ready",
            {"pulse_name": self.name, "global_time": global_time, "local_time": local_time, "next_fire": self.next_fire},
            "INFO",
            "PULSE_FIRING" if should_fire else "PULSE_NOT_READY"
        )
        return should_fire

    def on_signal(self, signal: Signal, state: 'State', ledger: TransparencyLedger) -> None:
        if signal.signature:
            try:
                verifying_key = nacl.signing.VerifyKey(
                    self.signing_key.verify_key.encode(encoder=nacl.encoding.RawEncoder)
                )
                verifying_key.verify(
                    f"{signal.from_agent}:{signal.name}:{signal.time}:{signal.strength}".encode('utf-8'),
                    signal.signature
                )
            except nacl.exceptions.BadSignatureError:
                self.logger.log(
                    "signal_verification",
                    signal.name.encode('utf-8'),
                    b"failed",
                    {"from_agent": signal.from_agent, "signal_time": signal.time, "strength": signal.strength},
                    "ERROR",
                    "INVALID_SIGNATURE"
                )
                return

        description = f"Received signal {signal.name} from {signal.from_agent} at time {signal.time} with strength {signal.strength}"
        self.logger.log(
            "signal_received",
            signal.name.encode('utf-8'),
            b"processed",
            {"from_agent": signal.from_agent, "signal_name": signal.name, "signal_time": signal.time, "strength": signal.strength},
            "INFO",
            "SUCCESS"
        )
        ledger.add_entry(
            self.name,
            "signal_received",
            {"from_agent": signal.from_agent, "signal_name": signal.name, "signal_time": signal.time, "strength": signal.strength},
            description
        )

        if signal.name == "sync":
            self.next_fire += signal.strength
            self.logger.log(
                "signal_sync",
                signal.name.encode('utf-8'),
                b"modulated",
                {"pulse_name": self.name, "next_fire": self.next_fire, "strength": signal.strength},
                "INFO",
                "SYNC_APPLIED"
            )
            print(f"🔄 Modulated [{self.name}] by +{signal.strength} at t={signal.time}")

        if signal.name == "off":
            self.enabled = False
            self.logger.log(
                "signal_off",
                signal.name.encode('utf-8'),
                b"disabled",
                {"pulse_name": self.name, "enabled": self.enabled, "strength": signal.strength},
                "INFO",
                "PULSE_DISABLED"
            )
            print(f"🛑 [{self.name}] switched OFF by signal at t={signal.time} (strength: {signal.strength})")

        if signal.name == "on":
            self.enabled = True
            self.logger.log(
                "signal_on",
                signal.name.encode('utf-8'),
                b"enabled",
                {"pulse_name": self.name, "enabled": self.enabled, "strength": signal.strength},
                "INFO",
                "PULSE_ENABLED"
            )
            print(f"✅ [{self.name}] switched ON by signal at t={signal.time} (strength: {signal.strength})")

    def fire(self, global_time: float, state: 'State', ledger: TransparencyLedger) -> List[Signal]:
        if not self.should_fire(global_time):
            return []

        if self.fractions <= 0:
            self.logger.log(
                "pulse_error",
                str(global_time).encode('utf-8'),
                b"invalid fractions",
                {"pulse_name": self.name, "fractions": self.fractions},
                "ERROR",
                "INVALID_FRACTIONS"
            )
            return []

        print(f"⏱️ Firing [{self.name}] at t={global_time}")
        self.logger.log(
            "pulse_fire",
            str(global_time).encode('utf-8'),
            b"fired",
            {"pulse_name": self.name, "global_time": global_time},
            "INFO",
            "PULSE_FIRED"
        )

        signals_emitted = []
        local_time = global_time * self.time_scale
        self.next_fire += self.interval

        for f in range(self.fractions):
            for action in self.body:
                if action.fraction is not None and action.fraction != f:
                    continue

                action_strength = min(1.0, max(0.0, action.strength if action.strength is not None else 1.0))
                if action.condition:
                    condition_met, condition_strength = self.evaluate_condition(action.condition)
                    if not condition_met:
                        self.logger.log(
                            "action_skip",
                            action.signal.encode('utf-8') if action.signal else b'',
                            b"condition not met",
                            {"pulse_name": self.name, "condition": action.condition, "strength": action_strength},
                            "INFO",
                            "CONDITION_FAIL"
                        )
                        continue
                    action_strength *= condition_strength

                action_details = {
                    "pulse_name": self.name,
                    "action_type": action.type,
                    "signal": action.signal,
                    "to": action.to,
                    "phase_shift": action.phase_shift,
                    "fraction": action.fraction,
                    "condition": action.condition,
                    "strength": action_strength
                }

                description = f"Performed action: {action.type} signal {action.signal} to {action.to} at time {global_time} with strength {action_strength}"
                ledger.add_entry(self.name, action.type, action_details, description)

                if action.type == "emit" and action.signal:
                    targets = [action.to] if isinstance(action.to, str) else (action.to or [])
                    for target in targets:
                        sig = Signal(
                            from_agent=self.name,
                            name=action.signal,
                            time=global_time,
                            strength=action_strength,
                            signature=self.sign_signal(action.signal, global_time, action_strength)
                        )
                        if target in state.pulses:
                            state.pulses[target].inbox.append(sig)
                            self.logger.log(
                                "action_emit",
                                action.signal.encode('utf-8'),
                                b"emitted",
                                {"target": target, "signal_name": action.signal, "strength": action_strength},
                                "INFO",
                                "ACTION_EMITTED"
                            )
                            print(f"📡 Emitted [{sig.name}] from [{self.name}] to [{target}] at t={global_time} (strength: {action_strength})")
                        signals_emitted.append(sig)
                        state.signals.append(sig)

                elif action.type == "broadcast" and action.signal:
                    for target in state.pulses:
                        if target != self.name:
                            sig = Signal(
                                from_agent=self.name,
                                name=action.signal,
                                time=global_time,
                                strength=action_strength,
                                signature=self.sign_signal(action.signal, global_time, action_strength)
                            )
                            state.pulses[target].inbox.append(sig)
                            self.logger.log(
                                "action_broadcast",
                                action.signal.encode('utf-8'),
                                b"broadcasted",
                                {"target": target, "signal_name": action.signal, "strength": action_strength},
                                "INFO",
                                "ACTION_BROADCASTED"
                            )
                            print(f"📡 Broadcasted [{action.signal}] from [{self.name}] to [{target}] at t={global_time} (strength: {action_strength})")
                            signals_emitted.append(sig)
                            state.signals.append(sig)

                elif action.type == "modulate" and action.signal and action.phase_shift is not None:
                    if action.to in state.pulses:
                        target_pulse = state.pulses[action.to]
                        target_pulse.next_fire += action.phase_shift * action_strength
                        self.logger.log(
                            "action_modulate",
                            action.signal.encode('utf-8'),
                            b"modulated",
                            {"target": action.to, "phase_shift": action.phase_shift, "strength": action_strength},
                            "INFO",
                            "ACTION_MODULATED"
                        )
                        print(f"🛠️ Modulated [{action.to}] by {action.phase_shift * action_strength} from [{self.name}] at t={global_time}")

                elif action.type == "modulate_time" and action.fraction is not None:
                    if action.fraction == 0:
                        self.logger.log(
                            "action_modulate_time_error",
                            str(self.time_scale).encode('utf-8'),
                            b"invalid fraction",
                            {"pulse_name": self.name, "fraction": action.fraction},
                            "ERROR",
                            "INVALID_FRACTION"
                        )
                        continue
                    old_time_scale = self.time_scale
                    self.time_scale = max(0.1, min(self.time_scale / action.fraction * action_strength, 10.0))
                    self.logger.log(
                        "action_modulate_time",
                        str(old_time_scale).encode('utf-8'),
                        str(self.time_scale).encode('utf-8'),
                        {"pulse_name": self.name, "fraction": action.fraction, "new_time_scale": self.time_scale, "strength": action_strength},
                        "INFO",
                        "ACTION_MODULATED_TIME"
                    )
                    print(f"⏳ [{self.name}] time_scale changed from {old_time_scale} to {self.time_scale} at t={global_time} (strength: {action_strength})")

        for signal in self.inbox:
            self.on_signal(signal, state, ledger)
        self.inbox.clear()

        return signals_emitted

    def evaluate_condition(self, condition: str) -> tuple[bool, float]:
        if self.bci_adapter:
            brain_state, strength = self.bci_adapter.interpret(self.bci_adapter.get_current_state())
            return brain_state == condition, strength
        return True, 1.0

    def sign_signal(self, signal_name: str, timestamp: float, strength: float) -> bytes:
        msg = f"{self.name}:{signal_name}:{timestamp}:{strength}".encode('utf-8')
        return self.signing_key.sign(msg).signature

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "interval": self.interval,
            "next_fire": self.next_fire,
            "body": [{"type": a.type, "signal": a.signal, "to": a.to, "phase_shift": a.phase_shift,
                      "fraction": a.fraction, "condition": a.condition, "strength": a.strength} for a in self.body],
            "time_scale": self.time_scale,
            "fractions": self.fractions,
            "enabled": self.enabled
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Pulse':
        return cls(
            name=data["name"],
            interval=data["interval"],
            next_fire=data["next_fire"],
            body=[Action(**action) for action in data["body"]],
            time_scale=data.get("time_scale", 1.0),
            fractions=data.get("fractions", 1),
            enabled=data.get("enabled", True)
        )

# === Quantum Pulse ===
class QuantumPulse(Pulse):
    def __init__(self, *args, state_vector: Optional[Dict[str, float]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_vector = state_vector or {"on": 0.5, "off": 0.5}

    def should_fire(self, global_time: float) -> bool:
        if not self.enabled:
            self.logger.log(
                "pulse_check",
                str(global_time).encode('utf-8'),
                b"disabled",
                {"pulse_name": self.name, "enabled": self.enabled},
                "INFO",
                "PULSE_DISABLED"
            )
            return False

        collapsed_state, strength = self.collapse_state()
        should_fire = collapsed_state == "on"
        self.logger.log(
            "pulse_check",
            str(global_time).encode('utf-8'),
            b"firing" if should_fire else b"not_firing",
            {"pulse_name": self.name, "global_time": global_time, "state": collapsed_state, "strength": strength},
            "INFO",
            "PULSE_FIRING" if should_fire else "PULSE_NOT_READY"
        )
        return should_fire

    def collapse_state(self) -> tuple[str, float]:
        rand = random.random()
        cumulative = 0.0
        for state, prob in self.state_vector.items():
            cumulative += prob
            if rand <= cumulative:
                strength = prob if state == "on" else 1.0 - prob
                return state, min(1.0, max(0.0, strength))
        return "off", 0.5

@dataclass
class State:
    time: float = 0.0
    pulses: Dict[str, Any] = field(default_factory=dict)
    signals: List[Signal] = field(default_factory=list)

    def save_state(self, filename: str) -> None:
        state_data = {
            "time": self.time,
            "pulses": {name: pulse.to_dict() for name, pulse in self.pulses.items()},
            "signals": [{"from_agent": s.from_agent, "name": s.name, "time": s.time, "strength": s.strength,
                         "signature": base64.b64encode(s.signature).decode('utf-8') if s.signature else None}
                        for s in self.signals]
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, indent=2)

    def load_state(self, filename: str) -> None:
        if not os.path.exists(filename):
            return
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.time = data.get("time", 0.0)
            self.pulses = {name: Pulse.from_dict(pulse_data) for name, pulse_data in data.get("pulses", {}).items()}
            self.signals = [
                Signal(
                    from_agent=s["from_agent"],
                    name=s["name"],
                    time=s["time"],
                    strength=s.get("strength", 1.0),
                    signature=base64.b64decode(s["signature"]) if s["signature"] else None
                ) for s in data.get("signals", [])
            ]

# === Pulse Configuration Loader ===
def load_pulse_config(filename: str) -> List[Pulse]:
    if not os.path.exists(filename):
        return []
    with open(filename, 'r', encoding='utf-8') as f:
        config = json.load(f)
        pulses = []
        for p in config.get("pulses", []):
            if p.get("is_quantum", False):
                pulses.append(QuantumPulse(
                    name=p["name"],
                    interval=p["interval"],
                    next_fire=p["next_fire"],
                    body=[Action(**action) for action in p["body"]],
                    time_scale=p.get("time_scale", 1.0),
                    fractions=p.get("fractions", 1),
                    enabled=p.get("enabled", True),
                    state_vector=p.get("state_vector", {"on": 0.5, "off": 0.5})
                ))
            else:
                pulses.append(Pulse(
                    name=p["name"],
                    interval=p["interval"],
                    next_fire=p["next_fire"],
                    body=[Action(**action) for action in p["body"]],
                    time_scale=p.get("time_scale", 1.0),
                    fractions=p.get("fractions", 1),
                    enabled=p.get("enabled", True)
                ))
        return pulses

# === Mock EEG Source ===
class MockEEGSource:
    def read_brainwave_data(self) -> Dict[str, Any]:
        return {
            "alpha": random.uniform(0.0, 1.0),
            "beta": random.uniform(0.0, 1.0)
        }

# === Example Usage ===
if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    ledger = TransparencyLedger()
    state = State()

    # Initialize BCI adapter with mock EEG source
    eeg_source = MockEEGSource()
    bci_adapter = BCIAdapter(eeg_source)

    # Load state if available
    state.load_state("state.json")
    ledger.load_ledger("ledger.json", clear_if_invalid=True)

    # Load pulse configuration from JSON file if available, else use default
    config_file = "pulse_config.json"
    if os.path.exists(config_file):
        pulses = load_pulse_config(config_file)
        for pulse in pulses:
            if any(action.condition for action in pulse.body):
                pulse.bci_adapter = bci_adapter
            state.pulses[pFIND_IN_QUOTEpulse.name] = pulse
    else:
        state.pulses["alpha"] = Pulse(
            name="alpha",
            interval=4.0,
            next_fire=4.0,
            fractions=2,
            body=[
                Action(type="emit", signal="sync", to="beta", fraction=0, condition="relaxed", strength=0.8),
                Action(type="modulate_time", fraction=2, to="gamma", strength=0.9)
            ],
            bci_adapter=bci_adapter
        )
        state.pulses["beta"] = QuantumPulse(
            name="beta",
            interval=3.0,
            next_fire=3.0,
            fractions=1,
            body=[
                Action(type="emit", signal="beta_ping", to="gamma", condition="focused", strength=0.7),
                Action(type="emit", signal="off", to="gamma", fraction=0, strength=1.0)
            ],
            state_vector={"on": 0.7, "off": 0.3},
            bci_adapter=bci_adapter
        )
        state.pulses["gamma"] = Pulse(
            name="gamma",
            interval=5.0,
            next_fire=5.0,
            fractions=1,
            body=[
                Action(type="broadcast", signal="gamma_wave", condition="neutral", strength=0.6)
            ],
            bci_adapter=bci_adapter
        )
        # Save default configuration
        config = {
            "pulses": [
                {**p.to_dict(), "is_quantum": isinstance(p, QuantumPulse), "state_vector": getattr(p, "state_vector", None)}
                for p in state.pulses.values()
            ]
        }
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

    state.time = 0.0
    while state.time <= 20.0:
        for name in list(state.pulses.keys()):
            pulse = state.pulses[name]
            signals = pulse.fire(state.time, state, ledger)
            for sig in signals:
                description = f"Signal {sig.name} emitted from {pulse.name} at time {sig.time} with strength {sig.strength}"
                ledger.add_entry(pulse.name, "fire", {"signal": sig.name, "time": sig.time, "strength": sig.strength}, description)
        state.time += 1.0

    # Save state and logs
    state.save_state("state.json")
    for pulse in state.pulses.values():
        pulse.logger.save_log(f"{pulse.name}_log.json")
    ledger.save_ledger("ledger.json")

    print("\n=== Ledger Verification ===")
    for entry in ledger.entries:
        print(f"Ledger entry for {entry.pulse_name} ({entry.operation}) verified: {ledger.verify_entry(entry)}")

    print("\n=== Final Signals ===")
    for sig in sorted(state.signals, key=lambda x: x.time):
        print(f"🟢 {sig.from_agent} -> {sig.name} at t={sig.time} (strength: {sig.strength})")
"""
