"""Audio -> original Transkun frontend/backbone -> CTC tokens -> **kern."""
import argparse
import json
from pathlib import Path

from .CheckpointA2S import load_checkpoint
from .DataScore import read_audio
from .Score import ScoreCodec


def main(argv=None):
    parser = argparse.ArgumentParser(description="Transcribe audio to **kern with Transkun A2S")
    parser.add_argument("audioPath")
    parser.add_argument("outPath", help=".krn or .kern (Humdrum **kern)")
    parser.add_argument("--weight", required=True, help="A2S checkpoint; AMT weights require A2S training first")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tokensOut", help="Optional JSON with raw tokens and reconstruction diagnostics")
    parser.add_argument("--strict", action="store_true", help="Reject malformed predictions instead of recovering records")
    args = parser.parse_args(argv)
    if Path(args.outPath).suffix.lower() not in {".krn", ".kern"}:
        parser.error("A2S output must use .krn or .kern (**kern)")
    model, _ = load_checkpoint(args.weight, args.device)
    model.eval()
    audio = read_audio(args.audioPath, model.fs, model.conf.scoreMaxAudioSeconds)
    tokens = model.transcribe_score(audio.unsqueeze(0).to(args.device))[0]
    codec = ScoreCodec(model.conf.scoreCodecVersion)
    diagnostics = codec.write(tokens, args.outPath, strict=args.strict)
    result = {"codec_version": codec.version, "output_format": "**kern", "tokens": tokens, "diagnostics": diagnostics}
    if args.tokensOut:
        path = Path(args.tokensOut)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": args.outPath, **diagnostics}, sort_keys=True))


if __name__ == "__main__":
    main()
