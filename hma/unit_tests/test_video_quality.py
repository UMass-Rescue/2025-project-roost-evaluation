"""Tests for hma.video.quality, using synthetic frames.

Real footage is replaced by frames of random coloured blocks plus fine noise (sharp, with
large-scale structure like a real scene, and with a sharpness that falls with blur much as
real frames do: about 1,500, then 55 at 0.4% blur and 10 at 0.8%) and blurred copies of
them; junk is built directly (flat colours, colour bars).
"""

import cv2
import numpy as np
import pytest

from hma.video.frames import SampledFrame
from hma.video.quality import (
    BARS,
    BLURRY,
    FLAT,
    FrameQuality,
    QualityConfig,
    QualityReport,
    assess_frame,
    measure_frame,
    select_frames,
)

H, W = 360, 640
RNG = np.random.default_rng(0)


def detailed(seed=0, block=40, noise=20):
    """A sharp frame with large-scale structure and fine detail, like a busy real scene."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, (H // block + 1, W // block + 1, 3), dtype=np.uint8)
    image = cv2.resize(base, (W + block, H + block), interpolation=cv2.INTER_NEAREST)[:H, :W]
    return np.clip(image.astype(np.float32) + rng.normal(0, noise, image.shape), 0, 255).astype(np.uint8)


def blurred(image, sigma_fraction):
    return cv2.GaussianBlur(image, (0, 0), sigma_fraction * image.shape[1])


def colour_bars():
    bars = np.zeros((H, W, 3), np.uint8)
    colours = [(192, 192, 192), (192, 192, 0), (0, 192, 192), (0, 192, 0), (192, 0, 192), (0, 0, 192), (192, 0, 0)]
    for i, colour in enumerate(colours):
        bars[:, i * W // 7 : (i + 1) * W // 7] = colour
    return bars


def bars_with_strip():
    """Colour bars with a bright strip added: still bars, but with a higher edge sharpness."""
    image = colour_bars()
    image[int(H * 0.75) :, : W // 2] = 255
    return image


def solid(value):
    return np.full((H, W, 3), value, np.uint8)


def frame(index, image):
    return SampledFrame(index=index, seconds=index / 10, image=image)


class TestAssessFrame:
    @pytest.mark.parametrize(
        "image",
        [solid(0), solid(255), solid(128), solid((0, 0, 255)), solid(16)],
        ids=["black", "white", "mid gray", "solid red", "near black"],
    )
    def test_flat_frames_are_junk(self, image):
        quality = assess_frame(image)
        assert quality.junk and FLAT in quality.reasons

    def test_noise_added_to_a_flat_frame_does_not_hide_it(self):
        noisy = np.clip(solid(0).astype(np.float32) + RNG.normal(0, 1.0, (H, W, 3)), 0, 255).astype(np.uint8)
        assert FLAT in assess_frame(noisy).reasons

    def test_colour_bars_are_junk_but_not_flat(self):
        quality = assess_frame(colour_bars())
        assert quality.reasons == (BARS,)
        assert quality.luma_std > 20 and quality.row_diff < 0.7

    def test_a_horizontal_gradient_counts_as_bars(self):
        gradient = np.tile(np.linspace(0, 255, W, dtype=np.uint8)[None, :, None], (H, 1, 3))
        assert BARS in assess_frame(gradient).reasons

    def test_a_detailed_frame_is_clean(self):
        quality = assess_frame(detailed())
        assert not quality.junk and quality.reasons == ()
        assert quality.luma_std > 30 and quality.sharpness > 500

    def test_a_heavily_blurred_frame_is_blurry_but_not_flat(self):
        quality = assess_frame(blurred(detailed(), 0.015))
        assert quality.reasons == (BLURRY,)

    def test_blur_lowers_the_sharpness_step_by_step(self):
        sharpness = [measure_frame(blurred(detailed(), f) if f else detailed())[2] for f in (0, 0.004, 0.008, 0.02)]
        assert sharpness == sorted(sharpness, reverse=True)

    def test_a_flat_frame_is_not_also_reported_blurry(self):
        assert assess_frame(solid(0)).reasons == (FLAT,)

    def test_measures_do_not_depend_on_resolution(self):
        small = detailed()
        large = cv2.resize(small, (W * 3, H * 3), interpolation=cv2.INTER_AREA)
        # the same picture at two resolutions lands on the same side of every threshold
        assert assess_frame(small).junk == assess_frame(large).junk == False  # noqa: E712

    def test_gray_input_is_accepted(self):
        gray = cv2.cvtColor(detailed(), cv2.COLOR_BGR2GRAY)
        assert not assess_frame(gray).junk

    def test_thresholds_are_configurable(self):
        image = blurred(detailed(), 0.004)
        assert not assess_frame(image, QualityConfig(blur_min=1.0)).junk
        assert BLURRY in assess_frame(image, QualityConfig(blur_min=10_000.0)).reasons


class TestQualityConfig:
    def test_defaults_sit_between_real_frames_and_junk(self):
        config = QualityConfig()
        assert (config.analysis_width, config.flat_std_max, config.bars_row_diff_max, config.blur_min) == (
            256,
            5.0,
            0.7,
            20.0,
        )

    @pytest.mark.parametrize("kwargs", [{"analysis_width": 4}, {"flat_std_max": -1}, {"blur_min": -0.1}])
    def test_rejects_invalid_values(self, kwargs):
        with pytest.raises(ValueError):
            QualityConfig(**kwargs)

    def test_from_env_reads_every_setting(self, monkeypatch):
        monkeypatch.setenv("VIDEO_QUALITY_ANALYSIS_WIDTH", "128")
        monkeypatch.setenv("VIDEO_FLAT_STD_MAX", "3")
        monkeypatch.setenv("VIDEO_BARS_ROW_DIFF_MAX", "0.5")
        monkeypatch.setenv("VIDEO_BLUR_MIN", "40")
        assert QualityConfig.from_env() == QualityConfig(128, 3.0, 0.5, 40.0)

    def test_from_env_keeps_defaults_when_unset_or_empty(self, monkeypatch):
        for name in ("VIDEO_QUALITY_ANALYSIS_WIDTH", "VIDEO_FLAT_STD_MAX", "VIDEO_BARS_ROW_DIFF_MAX"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("VIDEO_BLUR_MIN", " ")
        assert QualityConfig.from_env() == QualityConfig()


class TestSelectFrames:
    def test_keeps_the_sharpest_usable_frame_of_each_group(self):
        sharp, soft = detailed(1), blurred(detailed(2), 0.004)
        groups = [[frame(1, soft), frame(2, sharp), frame(3, soft)], [frame(11, detailed(3))]]
        selected, report = select_frames(groups)
        assert [s.frame.index for s in selected] == [2, 11]
        assert report.slices == 2 and report.slices_dropped == 0 and report.frames_assessed == 4

    def test_prefers_a_clean_frame_over_a_sharper_junk_one(self):
        junk, soft = bars_with_strip(), blurred(detailed(), 0.004)
        assert assess_frame(junk).junk and not assess_frame(soft).junk
        assert assess_frame(junk).sharpness > assess_frame(soft).sharpness  # the junk frame is the sharper one
        selected, _ = select_frames([[frame(1, junk), frame(2, soft)]])
        assert [s.frame.index for s in selected] == [2]

    def test_a_group_with_only_junk_is_dropped_and_counted(self):
        groups = [[frame(1, detailed())], [frame(10, solid(0)), frame(11, colour_bars())], [frame(20, detailed(5))]]
        selected, report = select_frames(groups)
        assert [s.frame.index for s in selected] == [1, 20]
        assert report.slices == 3 and report.slices_dropped == 1
        assert report.junk_frames == {FLAT: 1, BARS: 1}
        assert not report.no_usable_frames

    def test_a_video_of_only_junk_yields_nothing_and_says_so(self):
        groups = [[frame(1, solid(0))], [frame(10, solid(255))], [frame(20, colour_bars())]]
        selected, report = select_frames(groups)
        assert selected == []
        assert report.no_usable_frames
        assert report.slices_dropped == report.slices == 3
        assert "NO USABLE FRAMES" in report.summary()

    def test_no_groups_is_not_reported_as_all_junk(self):
        selected, report = select_frames([])
        assert selected == [] and not report.no_usable_frames
        assert "NO USABLE FRAMES" not in report.summary()

    def test_the_report_summary_names_the_reasons(self):
        _, report = select_frames([[frame(1, detailed())], [frame(10, solid(0))]])
        text = report.summary()
        assert "2 slices" in text and "1 flat" in text and "1 slices dropped" in text

    def test_groups_are_consumed_one_at_a_time(self):
        seen = []

        def groups():
            for i in range(3):
                seen.append(i)
                yield [frame(i, detailed(i))]

        selected, _ = select_frames(groups())
        assert seen == [0, 1, 2] and len(selected) == 3

    def test_selected_frames_carry_their_quality(self):
        selected, _ = select_frames([[frame(1, detailed())]])
        assert isinstance(selected[0].quality, FrameQuality) and not selected[0].quality.junk

    def test_an_empty_group_counts_as_a_dropped_slice(self):
        selected, report = select_frames([[], [frame(10, detailed())]])
        assert len(selected) == 1 and report.slices == 2 and report.slices_dropped == 1
