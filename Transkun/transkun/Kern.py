"""Direct **kern IO with exact rhythm and variable Humdrum spine paths.

music21 supplies note/tandem conversion, not file-level timing or export.
No MusicXML or MIDI intermediate is used.
"""
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from fractions import Fraction
import re

from music21 import chord, clef, humdrum, key, meter, note, stream, tempo


def walk(container, offset=0, voice_id=None):
    """Traverse notation without losing ownership or offsets when flattening."""
    for element in container:
        at = Fraction(offset) + Fraction(element.offset)
        if isinstance(element, stream.Stream):
            child = str(element.id) if isinstance(element, stream.Voice) else voice_id
            yield from walk(element, at, child)
        else:
            yield element, at, voice_id


def reciprocal(duration):
    duration = Fraction(duration)
    if duration <= 0:
        raise ValueError("Kern duration must be positive")
    for dots in range(4):
        base = duration / (2 - Fraction(1, 2 ** dots))
        if base >= 8 and (base / 4).denominator == 1 and (base.numerator // 4).bit_count() == 1:
            return "0" * ((base.numerator // 4).bit_length() - 1) + "." * dots
        value = 4 / base
        if value.denominator == 1:
            return str(value.numerator) + "." * dots
    value = 4 / duration
    return f"{value.numerator}%{value.denominator}"


def _note_token(single, duration, tie_type=None):
    p = single.pitch
    if p.octave is None or not float(p.ps).is_integer():
        raise ValueError("Kern output requires octave and integral semitone pitch")
    name = p.step.lower() * (p.octave - 3) if p.octave >= 4 else p.step * (4 - p.octave)
    alter = int(p.accidental.alter) if p.accidental else 0
    accidental = "#" * alter if alter > 0 else "-" * (-alter)
    body = reciprocal(duration) + name + accidental
    if tie_type == "start":
        return "[" + body
    if tie_type == "stop":
        return body + "]"
    if tie_type == "continue":
        return body + "_"
    return body


def _metadata_token(obj):
    if isinstance(obj, meter.TimeSignature):
        return "*M" + obj.ratioString
    if isinstance(obj, key.KeySignature):
        letters = "fcgdaeb" if obj.sharps >= 0 else "beadgcf"
        accidental = "#" if obj.sharps >= 0 else "-"
        return "*k[" + "".join(c + accidental for c in letters[:abs(obj.sharps)]) + "]"
    if isinstance(obj, clef.Clef):
        octaves = "^" * max(obj.octaveChange, 0) + "v" * max(-obj.octaveChange, 0)
        return f"*clef{obj.sign}{octaves}{obj.line}"
    if isinstance(obj, tempo.MetronomeMark) and obj.getQuarterBPM() is not None:
        return f"*MM{float(obj.getQuarterBPM()):.12g}"
    return None


def dumps_kern(score):
    """Serialize non-overlapping voices as synchronized, full-length spines.

    Gaps become rests; '.' means an event is still sounding. Notes crossing
    metadata/barline boundaries are split with per-pitch ties. The number of
    spines is inferred independently for each score, never fixed by the model.
    """
    columns, metadata = {}, {}
    parts = list(score.parts) if isinstance(score, stream.Score) else [score]
    part_meters = {}
    for part_index, part in enumerate(parts):
        for obj, at, voice in walk(part):
            if isinstance(obj, (note.Note, chord.Chord, note.Rest)):
                column = (part_index, voice if voice is not None else "0")
                columns.setdefault(column, []).append((at, at + Fraction(obj.quarterLength), obj))
            else:
                token = _metadata_token(obj)
                if token is not None:
                    metadata.setdefault(at, []).append((part_index, token))
                if isinstance(obj, meter.TimeSignature):
                    part_meters.setdefault(part_index, []).append((at, Fraction(obj.barDuration.quarterLength)))
    if not columns:
        raise ValueError("Kern score contains no events")
    order = sorted(columns, key=lambda item: (item[0], int(item[1])))
    end = max(b for events in columns.values() for _, b, _ in events)
    boundaries = {Fraction(0), end, *metadata.keys()}
    barlines = {Fraction(0), end}
    for part_index in range(len(parts)):
        changes = sorted(part_meters.get(part_index, [(Fraction(0), Fraction(4))]))
        if changes[0][0] != 0:
            changes.insert(0, (Fraction(0), Fraction(4)))
        for index, (at, size) in enumerate(changes):
            limit = changes[index + 1][0] if index + 1 < len(changes) else end
            if size <= 0:
                raise ValueError("Invalid Kern meter")
            while at < limit:
                barlines.add(at)
                at += size
    boundaries.update(barlines)
    cuts = sorted(boundaries)
    rows = {}
    for column in order:
        events = sorted(columns[column], key=lambda event: event[:2])
        complete = []
        cursor = Fraction(0)
        for begin, finish, obj in events:
            if begin < cursor or finish <= begin:
                raise ValueError("Overlapping/invalid Kern voice; normalize score voices first")
            if begin > cursor:
                complete.append((cursor, begin, note.Rest()))
            complete.append((begin, finish, obj))
            cursor = finish
        if cursor < end:
            complete.append((cursor, end, note.Rest()))
        for begin, finish, obj in complete:
            splits = [begin] + cuts[bisect_right(cuts, begin):bisect_left(cuts, finish)] + [finish]
            for index, (a, b) in enumerate(zip(splits, splits[1:])):
                if isinstance(obj, note.Rest):
                    token = reciprocal(b - a) + "r"
                else:
                    singles = obj.notes if isinstance(obj, chord.Chord) else [obj]
                    tokens = []
                    for single in singles:
                        original = single.tie.type if single.tie else "none"
                        incoming = index > 0 or original in {"continue", "stop"}
                        outgoing = index < len(splits) - 2 or original in {"start", "continue"}
                        tie_type = "continue" if incoming and outgoing else "stop" if incoming else "start" if outgoing else None
                        tokens.append(_note_token(single, b - a, tie_type))
                    token = " ".join(tokens)
                rows.setdefault(a, {})[column] = token
    lines = ["\t".join(["**kern"] * len(order)),
             "\t".join(f"*part{part + 1}" for part, _ in order),
             "\t".join(f"*voice{voice}" for _, voice in order)]
    for at in sorted(set(rows) | set(metadata) | barlines):
        for part_index, token in sorted(set(metadata.get(at, []))):
            lines.append("\t".join(token if part == part_index else "*" for part, _ in order))
        if at in barlines:
            lines.append("\t".join(["==" if at == end else "=" + str(sum(t <= at for t in barlines))] * len(order)))
        if at in rows:
            lines.append("\t".join(rows[at].get(column, ".") for column in order))
    lines.append("\t".join(["*-"] * len(order)))
    return "\n".join(lines) + "\n"


@dataclass
class _Spine:
    uid: int
    root: int
    kind: str | None
    end: Fraction


def _event(token):
    if "q" in token or "Q" in token:
        raise ValueError("Grace notes are not supported")
    singles = token.split(" ")
    first = re.search(r"\d+(?:%\d+)?\.*", singles[0])
    if first is None:
        raise ValueError(f"Missing Kern duration: {token}")
    notes = []
    for subtoken in singles:
        if re.search(r"\d", subtoken) is None:
            subtoken = first.group() + subtoken
        notes.append(humdrum.spineParser.hdStringToNote(subtoken))
    duration = Fraction(notes[0].quarterLength)
    if duration <= 0 or any(Fraction(n.quarterLength) != duration for n in notes):
        raise ValueError("Kern chord must have one positive duration")
    if len(notes) == 1:
        return notes[0]
    if any(not isinstance(n, note.Note) for n in notes):
        raise ValueError("Invalid Kern chord")
    obj = chord.Chord(notes)
    obj.quarterLength = duration
    return obj


def loads_kern(text):
    """Read fixed/open polyphony, including *^/*v/*+/*-/*x and auxiliary spines.

    A shared rational clock preserves asynchronous onsets and the start of
    newly added spines (music21's file-level parser resets some such offsets).
    """
    active, entries, root_parts = [], [], {}
    clock, next_uid = Fraction(0), 0
    saw_kern = False

    def new_spine(kind=None, root=None, end=None):
        nonlocal next_uid
        uid = next_uid
        next_uid += 1
        return _Spine(uid, uid if root is None else root, kind, clock if end is None else end)

    for line_number, raw in enumerate(text.lstrip("\ufeff").splitlines(), 1):
        if not raw.strip() or raw.startswith("!!"):
            continue
        fields = raw.split("\t")
        try:
            if not active:
                if not all(value.startswith("**") for value in fields):
                    raise ValueError("Expected exclusive interpretations before data")
                active = [new_spine(value) for value in fields]
                saw_kern |= "**kern" in fields
                continue
            if len(fields) != len(active):
                raise ValueError(f"Expected {len(active)} spine fields, got {len(fields)}")
            if fields[0].startswith("!"):
                continue
            if fields[0].startswith("*"):
                if not all(value.startswith("*") for value in fields):
                    raise ValueError("Mixed interpretation/data record")
                exchanges = [i for i, value in enumerate(fields) if value == "*x"]
                if len(exchanges) % 2:
                    raise ValueError("Spine exchange requires pairs")
                for a, b in zip(exchanges[::2], exchanges[1::2]):
                    active[a], active[b] = active[b], active[a]
                updated, index = [], 0
                while index < len(active):
                    state, value = active[index], fields[index]
                    if value == "*v":
                        finish = index
                        while finish < len(active) and fields[finish] == "*v":
                            finish += 1
                        if finish - index < 2 or len({s.kind for s in active[index:finish]}) != 1:
                            raise ValueError("Invalid spine join")
                        state.end = max(s.end for s in active[index:finish])
                        updated.append(state)
                        index = finish
                        continue
                    if value == "*^":
                        # The new lane may enter while the original note sustains.
                        # Its first event must not inherit that lane's busy time.
                        updated.extend([state, new_spine(state.kind, state.root)])
                    elif value == "*+":
                        updated.extend([state, new_spine()])
                    elif value != "*-":
                        if value.startswith("**"):
                            state.kind = value
                            saw_kern |= value == "**kern"
                        elif value.startswith("*part"):
                            root_parts[state.root] = value[5:]
                        elif state.kind == "**kern" and value.startswith(("*M", "*k[", "*clef")):
                            match = re.fullmatch(r"\*clef([GFC])([v^]*)([1-5])", value)
                            if match:
                                obj = clef.Clef()
                                obj.sign, obj.line = match[1], int(match[3])
                                obj.octaveChange = match[2].count("^") - match[2].count("v")
                            else:
                                obj = humdrum.spineParser.kernTandemToObject(value)
                            if isinstance(obj, (clef.Clef, key.KeySignature, meter.TimeSignature, tempo.MetronomeMark)):
                                entries.append((state.root, clock, None, obj))
                        updated.append(state)
                    index += 1
                active = updated
                continue
            if fields[0].startswith("="):
                continue
            had_event = False
            for state, value in zip(active, fields):
                if state.kind != "**kern" or value == ".":
                    continue
                if state.end > clock:
                    raise ValueError("New event overlaps a sounding event in one spine")
                obj = _event(value)
                entries.append((state.root, clock, state.uid, obj))
                state.end = clock + Fraction(obj.quarterLength)
                had_event = True
            if had_event:
                clock = min(s.end for s in active if s.kind == "**kern" and s.end > clock)
        except Exception as exc:
            raise ValueError(f"Kern line {line_number}: {exc}") from exc
    if not saw_kern or not any(voice is not None for _, _, voice, _ in entries):
        raise ValueError("Kern input contains no pitched/rest events")
    if active:
        raise ValueError("Kern input has unterminated spines (expected *-)")
    score, parts, voices, seen_metadata = stream.Score(), {}, {}, set()
    for root, at, voice, obj in entries:
        part_id = ("part", root_parts[root]) if root in root_parts else ("root", root)
        if part_id not in parts:
            parts[part_id] = stream.Part()
            score.insert(0, parts[part_id])
        part = parts[part_id]
        if voice is None:
            signature = (part_id, at, _metadata_token(obj))
            if signature not in seen_metadata:
                part.insert(at, obj)
                seen_metadata.add(signature)
        else:
            lane = (part_id, voice)
            if lane not in voices:
                voices[lane] = stream.Voice(id=str(voice))
                part.insert(0, voices[lane])
            voices[lane].insert(at, obj)
    return score
