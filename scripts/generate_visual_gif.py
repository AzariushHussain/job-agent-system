# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow>=11", "playwright>=1.50"]
# ///
"""Render the editable architecture SVG into the README's portable GIF hero."""

from io import BytesIO
import os
from pathlib import Path
import shutil

from PIL import Image
from playwright.sync_api import sync_playwright


def find_browser() -> str | None:
    for command in ("google-chrome", "chromium", "chromium-browser", "msedge"):
        if executable := shutil.which(command):
            return executable
    for variable in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
        if directory := os.environ.get(variable):
            for suffix in (
                "Google/Chrome/Application/chrome.exe",
                "Microsoft/Edge/Application/msedge.exe",
            ):
                executable = Path(directory) / suffix
                if executable.is_file():
                    return str(executable)
    return None  # Playwright's installed Chromium is the remaining option.


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "assets/animated-job-agent-flow.svg"
    output = root / "assets/infrastructure_and_evals.gif"
    frames = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=find_browser(), headless=True
        )
        try:
            page = browser.new_page(
                viewport={"width": 1200, "height": 900},
                reduced_motion="no-preference",
            )
            page.set_content(
                '<body style="margin:0;background:#090d16">'
                + source.read_text(encoding="utf-8")
                + "</body>"
            )
            svg = page.locator("svg")
            svg.evaluate("e => { e.setAttribute('width', '1200'); e.setAttribute('height', '900'); e.pauseAnimations(); }")
            for index in range(48):
                svg.evaluate(
                    """(e, time) => {
                        e.setCurrentTime(time);
                        for (const animation of e.getAnimations({subtree:true})) {
                            animation.pause();
                            animation.currentTime = time * 1000;
                        }
                    }""",
                    index / 10,
                )
                frames.append(Image.open(BytesIO(svg.screenshot())).convert("RGB"))
        finally:
            browser.close()

    palette = frames[0].quantize(colors=128)
    frames = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    frames[0].save(
        output, save_all=True, append_images=frames[1:], duration=100, loop=0
    )
    with Image.open(output) as result:
        assert result.size == (1200, 900) and result.n_frames > 1
    print(f"Rendered {output.name}: {len(frames)} frames, {output.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
