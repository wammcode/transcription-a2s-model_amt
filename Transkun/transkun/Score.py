"""Versioned, time-ordered score tokens and direct **kern reconstruction.

Positions/durations are rational quarter lengths, never performance seconds.
No MIDI is generated or quantized in the A2S path.
"""
from fractions import Fraction
from pathlib import Path

from music21 import chord, clef, converter, key, meter, note, pitch, stream, tempo, tie

from .Kern import dumps_kern, loads_kern, walk

LEGACY_CODEC_VERSION = "transkun-score-v1"
CODEC_VERSION = "transkun-score-v2"
SUPPORTED_CODECS = {LEGACY_CODEC_VERSION, CODEC_VERSION}
BLANK = "<BLANK>"
VALUE_FIELDS = {"AT", "PART", "VOICE", "DUR", "KEY", "TEMPO", "METER"}
VALUE_CHARACTERS = "0123456789/-+"


def schema_vocabulary():
    """Corpus-independent notation alphabet, including compositional numbers."""
    tokens = {"END", "REST", "VALUE_END", *VALUE_FIELDS}
    tokens.update("NUM_" + char for char in VALUE_CHARACTERS)
    tokens.update("TIE_" + value for value in ["none", "start", "continue", "stop"])
    tokens.update(f"CLEF_{sign}{line}:{octave}" for sign in "GFC"
                  for line in range(1, 6) for octave in range(-2, 3))
    for octave in range(-1, 10):
        for step in "CDEFGAB":
            for alteration in range(-4, 5):
                p = pitch.Pitch(f"{step}{octave}")
                p.accidental = pitch.Accidental(alteration)
                if 0 <= p.ps <= 127:
                    tokens.add("PITCH_" + p.nameWithOctave)
    return tokens


def rational(value):
    result = Fraction(value).limit_denominator(1920)
    if abs(float(result) - float(value)) > 1e-7:
        raise ValueError("Score time cannot be represented with denominator <= 1920")
    return str(result)


class Vocabulary:
    def __init__(self, tokens):
        self.tokens = list(tokens)
        if not self.tokens or self.tokens[0] != BLANK:
            raise ValueError("Vocabulary index 0 must be <BLANK> (CTC blank/padding)")
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError("Duplicate vocabulary tokens")
        self.indices = {word: i for i, word in enumerate(self.tokens)}

    @classmethod
    def build(cls, sequences, codec_version=CODEC_VERSION):
        words = {word for sequence in sequences for word in sequence}
        if codec_version == CODEC_VERSION:
            words.update(schema_vocabulary())
        return cls([BLANK] + sorted(words - {BLANK}))

    def encode(self, sequence):
        unknown = sorted(set(sequence) - self.indices.keys())
        if unknown:
            raise ValueError(f"Score tokens absent from training vocabulary: {unknown[:10]}")
        if BLANK in sequence:
            raise ValueError("CTC blank cannot occur in score targets")
        return [self.indices[word] for word in sequence]

    def decode(self, indices):
        return [self.tokens[i] for i in indices if i != 0]

    def __len__(self):
        return len(self.tokens)


class ScoreCodec:
    version = CODEC_VERSION

    def __init__(self, version=CODEC_VERSION):
        if version not in SUPPORTED_CODECS:
            raise ValueError(f"Unsupported score codec: {version}")
        self.version = version

    def encode_file(self, path):
        if Path(path).suffix.lower() in {".mid", ".midi"}:
            raise ValueError("A2S requires notated score targets, not performance MIDI")
        if Path(path).suffix.lower() in {".krn", ".kern"}:
            return self.encode_kern(Path(path).read_text(encoding="utf-8"))
        return self.encode(converter.parse(str(path)))

    def encode_kern(self, text):
        return self.encode(loads_kern(text))

    def _serialize(self, tokens):
        if self.version == LEGACY_CODEC_VERSION:
            return tokens
        result = []
        for token in tokens:
            kind, _, value = token.partition("_")
            if kind in VALUE_FIELDS:
                if not value or any(char not in VALUE_CHARACTERS for char in value):
                    raise ValueError(f"Unsupported numeric score field: {token}")
                result.extend([kind, *("NUM_" + char for char in value), "VALUE_END"])
            else:
                result.append(token)
        return result

    def _record(self, tokens):
        """Collapse v2 numeric fields for semantic validation; v1 stays readable."""
        result, index = [], 0
        while index < len(tokens):
            kind = tokens[index]
            if kind in VALUE_FIELDS:
                index += 1
                digits = []
                while index < len(tokens) and tokens[index] != "VALUE_END":
                    token = tokens[index]
                    if not token.startswith("NUM_") or token[4:] not in VALUE_CHARACTERS or len(token[4:]) != 1:
                        raise ValueError("Invalid compositional number")
                    digits.append(token[4:])
                    index += 1
                if index == len(tokens) or not digits:
                    raise ValueError("Unterminated/empty score number")
                result.append(kind + "_" + "".join(digits))
            else:
                result.append(kind)
            index += 1
        return result

    @staticmethod
    def _normalize_voices(items, legacy=False):
        """Group simultaneous equal-duration pitches and color overlapping lanes.

        Declared voice groups never merge with each other. Without voice labels,
        a deterministic minimum-lane allocation admits arbitrary open polyphony.
        """
        metadata, groups = [], {}
        base_indices = {None: 0} if legacy else {}
        for obj, at, source in items:
            if not isinstance(obj, (note.Note, chord.Chord, note.Rest)):
                metadata.append((obj, at, source))
                continue
            if source not in base_indices:
                base_indices[source] = len(base_indices)
            group_key = (source, at, Fraction(obj.quarterLength))
            candidates = groups.setdefault(group_key, [])
            obj_notes = obj.notes if isinstance(obj, chord.Chord) else [obj] if isinstance(obj, note.Note) else []
            for index, candidate in enumerate(candidates):
                other = candidate.notes if isinstance(candidate, chord.Chord) else [candidate] if isinstance(candidate, note.Note) else []
                if obj_notes and other and not ({n.pitch.nameWithOctave for n in obj_notes} & {n.pitch.nameWithOctave for n in other}):
                    merged = chord.Chord([*other, *obj_notes])
                    merged.quarterLength = obj.quarterLength
                    candidates[index] = merged
                    break
            else:
                candidates.append(obj)
        lane_ends, lane_indices = {}, {}
        next_index = len(base_indices)
        events = [(obj, at, source) for (source, at, _), objs in groups.items() for obj in objs]
        events.sort(key=lambda item: (item[1], base_indices[item[2]], Fraction(item[0].quarterLength)))
        for obj, at, source in events:
            ends = lane_ends.setdefault(source, [])
            lane = next((i for i, end in enumerate(ends) if end <= at), len(ends))
            if lane == len(ends):
                ends.append(at)
                if lane == 0:
                    lane_indices[source, lane] = base_indices[source]
                else:
                    lane_indices[source, lane] = next_index
                    next_index += 1
            ends[lane] = at + Fraction(obj.quarterLength)
            metadata.append((obj, at, lane_indices[source, lane]))
        return metadata

    def encode(self, score):
        parts = list(score.parts) if isinstance(score, stream.Score) else [score]
        records = []

        for part_index, part in enumerate(parts):
            metadata = {"meter": False, "key": False, "clef": False}
            items = self._normalize_voices(walk(part), legacy=self.version == LEGACY_CODEC_VERSION)
            for element, offset, voice_id in items:
                at = rational(offset)
                prefix = [f"AT_{at}", f"PART_{part_index}"]
                body = None
                priority = 0
                if isinstance(element, meter.TimeSignature):
                    body = [f"METER_{element.ratioString}"]
                    metadata["meter"] |= offset == 0
                elif isinstance(element, key.KeySignature):
                    body = [f"KEY_{element.sharps}"]
                    metadata["key"] |= offset == 0
                elif isinstance(element, clef.Clef):
                    if element.sign not in {"G", "F", "C"} or element.line is None:
                        raise ValueError("Only pitched G/F/C clefs are supported")
                    body = [f"CLEF_{element.sign}{element.line}:{element.octaveChange}"]
                    metadata["clef"] |= offset == 0
                elif isinstance(element, tempo.MetronomeMark):
                    bpm = element.getQuarterBPM()
                    if bpm is not None:
                        body = [f"TEMPO_{rational(bpm)}"]
                elif isinstance(element, (note.Note, chord.Chord, note.Rest)):
                    priority = 1
                    if element.duration.isGrace or element.quarterLength <= 0:
                        raise ValueError("Grace notes/zero-duration notes are not supported")
                    body = [f"VOICE_{voice_id}", f"DUR_{rational(element.quarterLength)}"]
                    if isinstance(element, note.Rest):
                        body.append("REST")
                    else:
                        notes = element.notes if isinstance(element, chord.Chord) else [element]
                        for single in sorted(notes, key=lambda n: (n.pitch.midi, n.pitch.nameWithOctave)):
                            if single.pitch.microtone.cents != 0 or not float(single.pitch.ps).is_integer():
                                raise ValueError("Microtonal pitches are not supported")
                            body.extend([f"PITCH_{single.pitch.nameWithOctave}",
                                         f"TIE_{single.tie.type if single.tie else 'none'}"])
                elif isinstance(element, note.Unpitched):
                    raise ValueError("Unpitched percussion is not supported")
                if body is not None:
                    records.append((Fraction(at), priority, part_index, prefix + body + ["END"]))
            for kind, default in [("meter", "METER_4/4"), ("key", "KEY_0"), ("clef", "CLEF_G2:0")]:
                if not metadata[kind]:
                    records.append((Fraction(0), 0, part_index,
                                    ["AT_0", f"PART_{part_index}", default, "END"]))
        # Keep simultaneous events stable, with explicit part/voice identity.
        records.sort(key=lambda item: (item[0], item[1], item[2], tuple(item[3])))
        return self._serialize([token for record in records for token in record[3]])

    def decode(self, tokens, strict=False):
        """Return (Score, diagnostics); skip malformed predicted records explicitly."""
        score = stream.Score()
        parts, voices, voice_intervals = {}, {}, {}
        diagnostics = {"records": 0, "rejected_records": 0, "events": 0, "fallback": False}

        def parse_record(record):
            record = self._record(record)
            if len(record) < 3 or not record[0].startswith("AT_") or not record[1].startswith("PART_"):
                raise ValueError("Record requires AT and PART")
            at = Fraction(record[0][3:])
            part_id = int(record[1][5:])
            if not 0 <= at <= 10000 or part_id < 0:
                raise ValueError("Position or part outside supported bounds")
            body = record[2:]
            voice_id = None
            if body[0].startswith("VOICE_"):
                voice_id = int(body[0][6:])
                if voice_id < 0 or len(body) < 3 or not body[1].startswith("DUR_"):
                    raise ValueError("Invalid voice/duration")
                duration = Fraction(body[1][4:])
                if not 0 < duration <= 256:
                    raise ValueError("Duration outside supported bounds")
                values = body[2:]
                if values == ["REST"]:
                    obj = note.Rest()
                else:
                    if len(values) % 2 or not values:
                        raise ValueError("Expected PITCH/TIE pairs")
                    singles = []
                    for i in range(0, len(values), 2):
                        if not values[i].startswith("PITCH_") or not values[i + 1].startswith("TIE_"):
                            raise ValueError("Expected PITCH/TIE pairs")
                        p = pitch.Pitch(values[i][6:])
                        if p.octave is None or not 0 <= p.midi <= 127 or not float(p.ps).is_integer():
                            raise ValueError("Pitch outside supported bounds")
                        single = note.Note(p)
                        tie_type = values[i + 1][4:]
                        if tie_type not in {"none", "start", "continue", "stop"}:
                            raise ValueError("Invalid tie")
                        if tie_type != "none":
                            single.tie = tie.Tie(tie_type)
                        singles.append(single)
                    obj = singles[0] if len(singles) == 1 else chord.Chord(singles)
                obj.quarterLength = duration
                if any(at < end and at + duration > begin
                       for begin, end in voice_intervals.get((part_id, voice_id), [])):
                    raise ValueError("Overlapping events in one voice; use a chord or separate voices")
            elif len(body) == 1 and body[0].startswith("METER_"):
                obj = meter.TimeSignature(body[0][6:])
                if obj.barDuration.quarterLength > 256:
                    raise ValueError("Meter outside supported bounds")
            elif len(body) == 1 and body[0].startswith("KEY_"):
                sharps = int(body[0][4:])
                if not -7 <= sharps <= 7:
                    raise ValueError("Key outside supported bounds")
                obj = key.KeySignature(sharps)
            elif len(body) == 1 and body[0].startswith("CLEF_"):
                spec, octave = body[0][5:].split(":")
                if len(spec) != 2 or spec[0] not in "GFC" or spec[1] not in "12345" or int(octave) not in {-2, -1, 0, 1, 2}:
                    raise ValueError("Invalid clef")
                obj = clef.Clef()
                obj.sign, obj.line, obj.octaveChange = spec[0], int(spec[1]), int(octave)
            elif len(body) == 1 and body[0].startswith("TEMPO_"):
                bpm = Fraction(body[0][6:])
                if not 0 < bpm <= 1000:
                    raise ValueError("Invalid tempo")
                obj = tempo.MetronomeMark(number=float(bpm))
            else:
                raise ValueError("Unknown score record")
            if part_id not in parts:
                parts[part_id] = stream.Part(id=f"P{part_id}")
            part = parts[part_id]
            if voice_id is None:
                part.insert(at, obj)
            else:
                voice_key = (part_id, voice_id)
                if voice_key not in voices:
                    voices[voice_key] = stream.Voice(id=str(voice_id))
                    part.insert(0, voices[voice_key])
                voices[voice_key].insert(at, obj)
                voice_intervals.setdefault(voice_key, []).append((at, at + duration))
                diagnostics["events"] += 1

        record = []
        for token in tokens:
            if token == "END":
                diagnostics["records"] += 1
                try:
                    parse_record(record)
                except Exception as exc:
                    if strict:
                        raise ValueError(f"Invalid score record: {record}") from exc
                    diagnostics["rejected_records"] += 1
                record = []
            else:
                record.append(token)
        if record:
            if strict:
                raise ValueError("Unterminated score record")
            diagnostics["rejected_records"] += 1
        if not parts:
            parts[0] = stream.Part(id="P0")
        if diagnostics["events"] == 0:
            if strict:
                raise ValueError("Score contains no notes or rests")
            # A syntactically writable score is not evidence of musical accuracy.
            parts[min(parts)].insert(0, note.Rest(quarterLength=4))
            diagnostics["fallback"] = True
        for part_id in sorted(parts):
            part = parts[part_id]
            flat = part.flatten()
            for cls, default in [(meter.TimeSignature, meter.TimeSignature("4/4")),
                                 (key.KeySignature, key.KeySignature(0)),
                                 (clef.Clef, clef.TrebleClef())]:
                if not any(isinstance(e, cls) and e.offset == 0 for e in flat):
                    part.insert(0, default)
            score.insert(0, part)
        return score, diagnostics

    def write(self, tokens, path, strict=False):
        score, diagnostics = self.decode(tokens, strict=strict)
        path = Path(path)
        if path.suffix.lower() not in {".krn", ".kern"}:
            raise ValueError("A2S output must be .krn or .kern (**kern)")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dumps_kern(score), encoding="utf-8")
        return diagnostics
