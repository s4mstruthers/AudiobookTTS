# AudiobookTTS

Turn an EPUB into a chaptered `.m4b` audiobook, narrated by an open-source text-to-speech model running locally on your Mac.

You pick the narrator by giving it a short recording of any voice. AudiobookTTS cleans up the book's text so the model reads it naturally, renders it chapter by chapter, and masters the result to the loudness standard used by Audible (ACX). It runs fully offline, either from the command line or in a local web UI.

> **Platform:** the models run on [MLX](https://github.com/ml-explore/mlx), so this needs a Mac with Apple Silicon (M1 or later).

---

## Features

- **EPUB parsing that copes with real-world files.** Chapters come from the table of contents, including books where several chapters share one HTML file. Cover pages, copyright notices and other front matter are dropped. Titles and authors mangled by Calibre ("*Title* by *Author*" stuffed into the title field) are repaired.
- **Voice cloning from any recording.** A 15-second clip from a LibriVox reading, a podcast or your own phone becomes a selectable narrator. The clip is trimmed, converted to mono and loudness-matched before use.
- **Text normalisation for narration:**
  - numbers, ordinals and dates are written out in words, in British order ("2nd July" → "second of July")
  - abbreviations are expanded ("Dr." → "Doctor")
  - small-caps text set in ALL CAPS is converted back to normal case, because autoregressive models stumble over it
  - Roman-numeral chapter titles are read as numbers ("Chapter XII" → "Chapter 12"), and footnote markers like `[12]` are removed
- **Pronunciation lexicon.** You teach it how to say a name once ("Kizhi" → "Kee-zhee") and every render and preview uses that spelling. It can also scan a book and list the unfamiliar names most likely to need one.
- **Consistent pacing.** Each voice's natural speaking rate is measured once and corrected to a target pace (160 words per minute by default) using a pitch-preserving time-stretch. Pauses between paragraphs and after chapter titles are configurable.
- **Mastering to ACX standards.** The finished book goes through a high-pass filter, de-esser, light compressor, two-pass EBU R128 loudness normalisation (target −19 LUFS) and a peak limiter (−3.3 dB).
- **Resumable jobs.** Each chapter is saved when it finishes, so an interrupted conversion picks up where it stopped. A job refuses to resume if the source EPUB has changed since it started.
- **Per-chapter voices.** A chapter can be read by a different narrator, which helps books with more than one point of view.
- **Chaptered output.** The `.m4b` file has chapter markers, title and author metadata, and the book's cover (or one you supply).
- **Local web UI.** Upload a book, trim voice clips on a waveform, audition voices on your book's own text, edit pronunciations, and follow progress live.

## How it works

```
EPUB ─► parse chapters ─► clean text ─► split into paragraphs ─► TTS per chunk
     ─► trim silence + insert pauses ─► pace correction ─► chapter FLAC (checkpoint)
     ─► concatenate ─► master (loudness, peaks) ─► AAC .m4b with chapters + cover
```

| Module | Role |
|---|---|
| `epub.py` | Reads the EPUB and splits it into chapters of plain text, using the table of contents |
| `textproc.py`, `numbers.py`, `pronunciation.py` | Normalise the text and break it into sentence-aware chunks |
| `engines/` | TTS back-ends behind one `TTSEngine` interface (see below) |
| `pacing.py` | Measures each voice's words per minute and works out the time-stretch needed |
| `pipeline.py` | Renders chapters, inserts pauses, saves a checkpoint per chapter, reports progress |
| `mastering.py`, `packaging.py` | Run the ffmpeg mastering chain and build the `.m4b` |
| `store.py` | Keeps one folder plus a JSON manifest per job, which is what makes resuming possible |
| `cli.py`, `web.py`, `static/` | The command-line interface, and the FastAPI server with the browser UI |

### TTS engines

| Engine | Model | Notes |
|---|---|---|
| `chatterbox` (default) | [Chatterbox Turbo](https://github.com/resemble-ai/chatterbox), 8-bit, via [mlx-audio](https://github.com/Blaizzy/mlx-audio) | About 0.5B parameters. Reproduces a speaker's timbre well. |
| `higgs` | Higgs Audio v2 3B, 8-bit, via mlx-audio | Larger and slower. May hold a non-native accent better. |

All model calls run on one dedicated thread. MLX GPU streams belong to the thread that created them, so this stops the web server's thread pool from crashing the process.

## Requirements

- macOS on Apple Silicon
- Python 3.10 or newer (developed on 3.13)
- [ffmpeg](https://ffmpeg.org/), including `ffprobe`: `brew install ffmpeg`

The model weights download from Hugging Face the first time you use an engine.

## Installation

```bash
git clone https://github.com/s4mstruthers/AudiobookTTS.git
cd AudiobookTTS
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
abtts --help
```

`pip install -e .` installs the Python dependencies and adds the `abtts` command. The `-e` (editable) flag means changes you make to the code take effect without reinstalling; drop it for a normal install.

## Usage

### Web UI

```bash
abtts serve                           # http://127.0.0.1:8765
```

### Command line

```bash
# 1. Make a narrator from any recording (the default start skips LibriVox intros)
abtts add-voice reading.mp3 --name narrator --start 45 --duration 15

# 2. Hear it on your own book before committing to a full render
abtts preview book.epub --voice narrator --chapter 3

# 3. Teach it difficult names
abtts say --suggest book.epub         # list names that probably need help
abtts say Kizhi "Kee-zhee"

# 4. Convert
abtts convert book.epub --voice narrator

# Interrupted? List jobs and resume one
abtts jobs
abtts resume <job_id>
```

| Command | What it does |
|---|---|
| `convert <epub>` | Converts an EPUB into an `.m4b` audiobook |
| `resume <job_id>` | Continues an interrupted job from its last finished chapter |
| `jobs` | Lists jobs with their status and progress |
| `voices [--engine]` | Lists the voices available to an engine |
| `preview <epub>` | Plays a short sample of a chapter in the chosen voice (uses macOS `afplay`) |
| `add-voice <audio> --name` | Turns a recording into a reference clip for cloning |
| `say [word] [respelling]` | Adds to, lists or removes entries in the pronunciation lexicon |
| `serve [--port]` | Starts the local web UI |

`convert`, `resume` and `preview` also accept `--engine`, `--voice`, `--speed`, `--bitrate` and `--output-dir`.

## Configuration

Settings live in `~/.audiobooktts/config.json`. Every key is optional.

| Key | Default | Meaning |
|---|---|---|
| `engine` | `"chatterbox"` | TTS engine |
| `target_wpm` | `160` | Narration pace that every voice is corrected to (`0` turns it off) |
| `synthesis_unit` | `"paragraph"` | `"paragraph"` reads naturally. `"sentence"` gives exact pauses but sounds choppier. |
| `gap_paragraph` / `gap_title` | `1.9` / `2.6` | Pause in seconds between paragraphs / after a chapter title |
| `master_audio` | `true` | Apply ACX mastering |
| `bitrate` | `"80k"` | AAC bitrate of the `.m4b` |
| `output_dir` | `~/Audiobooks` | Where finished books are saved |
| `chatterbox_model` | `mlx-community/chatterbox-turbo-8bit` | Chatterbox weights to load |

The same folder also holds:

- `voices/`: reference clips; the file name is the voice name
- `pronunciation.json`: the pronunciation lexicon
- `voice_pacing.json`: the measured speaking rate of each voice
- `jobs/`: one folder per conversion, with checkpointed chapter audio

## Notes

- **Use voice recordings responsibly.** Only clone voices you have the right to use. Public-domain LibriVox recordings are a good source.
- **Speed:** rendering a full novel takes hours, depending on the engine and your machine.
- **Personal use:** convert only books you are entitled to use.

## License

[MIT](LICENSE) © 2026 Sam Struthers
