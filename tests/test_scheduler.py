# coding=utf-8
"""Tests for trendradar.core.scheduler – focuses on half-open interval semantics."""

from datetime import datetime
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from trendradar.core.scheduler import Scheduler, ResolvedSchedule


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _minimal_timeline(
    periods: Dict[str, Any] | None = None,
    day_plans: Dict[str, Any] | None = None,
    week_map: Dict[int, str] | None = None,
    default: Dict[str, Any] | None = None,
    overlap_policy: str = "error_on_overlap",
) -> Dict[str, Any]:
    """Build a minimal but valid timeline dict."""
    if default is None:
        default = {"collect": True, "analyze": False, "push": False,
                    "report_mode": "current", "ai_mode": "follow_report"}
    if periods is None:
        periods = {}
    if day_plans is None:
        day_plans = {"workday": {"periods": list(periods.keys())}}
    if week_map is None:
        week_map = {d: "workday" for d in range(1, 8)}
    tl: Dict[str, Any] = {
        "default": default,
        "periods": periods,
        "day_plans": day_plans,
        "week_map": week_map,
        "overlap": {"policy": overlap_policy},
    }
    return tl


def _make_scheduler(
    timeline: Dict[str, Any],
    now: datetime | None = None,
    enabled: bool = True,
) -> Scheduler:
    """Construct a Scheduler with fakes for storage / time."""
    if now is None:
        now = datetime(2026, 6, 1, 10, 0, 0)  # Sunday 10:00

    storage = MagicMock()
    storage.has_period_executed = MagicMock(return_value=False)
    storage.record_period_execution = MagicMock()

    schedule_config: Dict[str, Any] = {"enabled": enabled, "preset": "custom"}
    timeline_data: Dict[str, Any] = {"custom": timeline}

    return Scheduler(
        schedule_config=schedule_config,
        timeline_data=timeline_data,
        storage_backend=storage,
        get_time_func=lambda: now,
    )


# ===================================================================
# _in_range  (static method – half-open [start, end))
# ===================================================================

class TestInRange:
    """Scheduler._in_range – half-open interval [start, end)."""

    # --- normal (non-cross-day) range ---

    def test_inside(self):
        assert Scheduler._in_range("10:00", "08:00", "18:00") is True

    def test_at_start(self):
        assert Scheduler._in_range("08:00", "08:00", "18:00") is True

    def test_at_end_excluded(self):
        """End boundary must be excluded in a half-open interval."""
        assert Scheduler._in_range("18:00", "08:00", "18:00") is False

    def test_before_start(self):
        assert Scheduler._in_range("07:59", "08:00", "18:00") is False

    def test_after_end(self):
        assert Scheduler._in_range("18:01", "08:00", "18:00") is False

    def test_one_minute_range(self):
        assert Scheduler._in_range("08:00", "08:00", "08:01") is True
        assert Scheduler._in_range("08:01", "08:00", "08:01") is False

    # --- cross-day range ---

    def test_cross_day_evening(self):
        assert Scheduler._in_range("23:00", "22:00", "07:00") is True

    def test_cross_day_morning(self):
        assert Scheduler._in_range("05:00", "22:00", "07:00") is True

    def test_cross_day_at_start(self):
        assert Scheduler._in_range("22:00", "22:00", "07:00") is True

    def test_cross_day_at_end_excluded(self):
        """End boundary excluded even for cross-day ranges."""
        assert Scheduler._in_range("07:00", "22:00", "07:00") is False

    def test_cross_day_outside_daytime(self):
        assert Scheduler._in_range("12:00", "22:00", "07:00") is False

    def test_cross_day_just_before_start(self):
        assert Scheduler._in_range("21:59", "22:00", "07:00") is False

    def test_cross_day_midnight(self):
        assert Scheduler._in_range("00:00", "22:00", "07:00") is True

    def test_cross_day_end_midnight(self):
        """Period 22:00-00:00: midnight is the end, so excluded."""
        assert Scheduler._in_range("00:00", "22:00", "00:00") is False
        assert Scheduler._in_range("23:59", "22:00", "00:00") is True


# ===================================================================
# _ranges_overlap  (static method – half-open semantics)
# ===================================================================

class TestRangesOverlap:
    """Scheduler._ranges_overlap – half-open interval overlap detection."""

    # --- non-cross-day ---

    def test_disjoint(self):
        assert Scheduler._ranges_overlap("08:00", "09:00", "10:00", "11:00") is False

    def test_adjacent_no_overlap(self):
        """Adjacent half-open intervals must NOT overlap: [08,09) ∩ [09,10) = ∅."""
        assert Scheduler._ranges_overlap("08:00", "09:00", "09:00", "10:00") is False

    def test_overlapping(self):
        assert Scheduler._ranges_overlap("08:00", "10:00", "09:00", "11:00") is True

    def test_one_inside_other(self):
        assert Scheduler._ranges_overlap("08:00", "18:00", "10:00", "12:00") is True

    def test_identical(self):
        assert Scheduler._ranges_overlap("08:00", "09:00", "08:00", "09:00") is True

    def test_overlap_by_one_minute(self):
        assert Scheduler._ranges_overlap("08:00", "09:01", "09:00", "10:00") is True

    # --- cross-day ---

    def test_cross_day_overlap(self):
        """22:00-07:00 overlaps with 06:00-08:00."""
        assert Scheduler._ranges_overlap("22:00", "07:00", "06:00", "08:00") is True

    def test_cross_day_no_overlap(self):
        """22:00-07:00 and 07:00-08:00 should NOT overlap (half-open)."""
        assert Scheduler._ranges_overlap("22:00", "07:00", "07:00", "08:00") is False

    def test_both_cross_day_overlap(self):
        """23:00-02:00 and 01:00-04:00 overlap in the morning."""
        assert Scheduler._ranges_overlap("23:00", "02:00", "01:00", "04:00") is True

    def test_cross_day_adjacent_at_midnight(self):
        """20:00-00:00 and 00:00-06:00 should NOT overlap (adjacent at midnight)."""
        assert Scheduler._ranges_overlap("20:00", "00:00", "00:00", "06:00") is False

    def test_cross_day_disjoint(self):
        """22:00-01:00 and 08:00-12:00 are completely disjoint."""
        assert Scheduler._ranges_overlap("22:00", "01:00", "08:00", "12:00") is False

    def test_both_cross_day_disjoint(self):
        """22:00-02:00 and 03:00-05:00 (treated as cross-day? no, 03<05). Disjoint."""
        assert Scheduler._ranges_overlap("22:00", "02:00", "03:00", "05:00") is False


# ===================================================================
# _validate_hhmm
# ===================================================================

class TestValidateHHMM:

    def test_valid(self):
        Scheduler._validate_hhmm("00:00", "test")
        Scheduler._validate_hhmm("23:59", "test")
        Scheduler._validate_hhmm("12:30", "test")

    def test_invalid_format(self):
        with pytest.raises(ValueError, match="格式错误"):
            Scheduler._validate_hhmm("8:00", "test")

    def test_hour_out_of_range(self):
        with pytest.raises(ValueError, match="超出范围"):
            Scheduler._validate_hhmm("25:00", "test")

    def test_minute_out_of_range(self):
        with pytest.raises(ValueError, match="超出范围"):
            Scheduler._validate_hhmm("12:60", "test")

    def test_garbage(self):
        with pytest.raises(ValueError, match="格式错误"):
            Scheduler._validate_hhmm("abc", "test")


# ===================================================================
# _validate_timeline
# ===================================================================

class TestValidateTimeline:

    def test_valid_minimal(self):
        tl = _minimal_timeline()
        _make_scheduler(tl)  # should not raise

    def test_missing_top_key(self):
        tl = _minimal_timeline()
        del tl["default"]
        with pytest.raises(ValueError, match="缺少必须字段"):
            _make_scheduler(tl)

    def test_missing_week_map_entry(self):
        tl = _minimal_timeline()
        del tl["week_map"][3]
        with pytest.raises(ValueError, match="week_map 缺少星期映射"):
            _make_scheduler(tl)

    def test_bad_day_plan_ref(self):
        tl = _minimal_timeline()
        tl["week_map"][1] = "nonexistent"
        with pytest.raises(ValueError, match="不存在的 day_plan"):
            _make_scheduler(tl)

    def test_bad_period_ref(self):
        tl = _minimal_timeline(
            day_plans={"workday": {"periods": ["missing_period"]}},
        )
        with pytest.raises(ValueError, match="不存在的 period"):
            _make_scheduler(tl)

    def test_period_missing_start_end(self):
        tl = _minimal_timeline(periods={"p1": {"name": "test"}})
        with pytest.raises(ValueError, match="缺少 start 或 end"):
            _make_scheduler(tl)

    def test_period_same_start_end(self):
        tl = _minimal_timeline(periods={"p1": {"start": "08:00", "end": "08:00"}})
        with pytest.raises(ValueError, match="start 与 end 不能相同"):
            _make_scheduler(tl)

    def test_period_bad_time_format(self):
        tl = _minimal_timeline(periods={"p1": {"start": "8:00", "end": "09:00"}})
        with pytest.raises(ValueError, match="格式错误"):
            _make_scheduler(tl)

    def test_adjacent_periods_no_overlap_error(self):
        """Adjacent half-open periods should NOT trigger overlap error."""
        tl = _minimal_timeline(
            periods={
                "morning": {"start": "08:00", "end": "12:00"},
                "afternoon": {"start": "12:00", "end": "18:00"},
            },
            overlap_policy="error_on_overlap",
        )
        _make_scheduler(tl)  # should not raise

    def test_overlapping_periods_raise(self):
        tl = _minimal_timeline(
            periods={
                "morning": {"start": "08:00", "end": "13:00"},
                "afternoon": {"start": "12:00", "end": "18:00"},
            },
            overlap_policy="error_on_overlap",
        )
        with pytest.raises(ValueError, match="存在重叠"):
            _make_scheduler(tl)

    def test_overlapping_periods_last_wins_ok(self):
        """Overlapping periods don't error when policy is last_wins."""
        tl = _minimal_timeline(
            periods={
                "morning": {"start": "08:00", "end": "13:00"},
                "afternoon": {"start": "12:00", "end": "18:00"},
            },
            overlap_policy="last_wins",
        )
        _make_scheduler(tl)  # should not raise

    def test_cross_day_adjacent_no_overlap(self):
        """22:00-07:00 and 07:00-12:00 are adjacent, not overlapping."""
        tl = _minimal_timeline(
            periods={
                "night": {"start": "22:00", "end": "07:00"},
                "morning": {"start": "07:00", "end": "12:00"},
            },
            overlap_policy="error_on_overlap",
        )
        _make_scheduler(tl)  # should not raise


# ===================================================================
# _find_active_period
# ===================================================================

class TestFindActivePeriod:

    def test_hit_period(self):
        tl = _minimal_timeline(periods={
            "morning": {"start": "08:00", "end": "12:00"},
        })
        s = _make_scheduler(tl, now=datetime(2026, 6, 1, 10, 0))
        result = s._find_active_period("10:00", {"periods": ["morning"]})
        assert result == "morning"

    def test_no_hit(self):
        tl = _minimal_timeline(periods={
            "morning": {"start": "08:00", "end": "12:00"},
        })
        s = _make_scheduler(tl, now=datetime(2026, 6, 1, 14, 0))
        result = s._find_active_period("14:00", {"periods": ["morning"]})
        assert result is None

    def test_boundary_end_excluded(self):
        """At the exact end time, period should NOT be active."""
        tl = _minimal_timeline(periods={
            "morning": {"start": "08:00", "end": "12:00"},
        })
        s = _make_scheduler(tl, now=datetime(2026, 6, 1, 12, 0))
        result = s._find_active_period("12:00", {"periods": ["morning"]})
        assert result is None

    def test_boundary_start_included(self):
        tl = _minimal_timeline(periods={
            "morning": {"start": "08:00", "end": "12:00"},
        })
        s = _make_scheduler(tl, now=datetime(2026, 6, 1, 8, 0))
        result = s._find_active_period("08:00", {"periods": ["morning"]})
        assert result == "morning"

    def test_adjacent_periods_boundary(self):
        """At boundary between two adjacent periods, only the second is active."""
        tl = _minimal_timeline(
            periods={
                "morning": {"start": "08:00", "end": "12:00"},
                "afternoon": {"start": "12:00", "end": "18:00"},
            },
        )
        s = _make_scheduler(tl, now=datetime(2026, 6, 1, 12, 0))
        result = s._find_active_period("12:00", {"periods": ["morning", "afternoon"]})
        assert result == "afternoon"

    def test_overlap_error_on_overlap(self):
        tl = _minimal_timeline(
            periods={
                "a": {"start": "08:00", "end": "13:00"},
                "b": {"start": "10:00", "end": "15:00"},
            },
            overlap_policy="last_wins",  # skip validation error
        )
        s = _make_scheduler(tl, now=datetime(2026, 6, 1, 11, 0))
        # Override overlap policy for runtime detection
        s.timeline["overlap"] = {"policy": "error_on_overlap"}
        with pytest.raises(ValueError, match="重叠冲突"):
            s._find_active_period("11:00", {"periods": ["a", "b"]})

    def test_overlap_last_wins(self):
        tl = _minimal_timeline(
            periods={
                "a": {"start": "08:00", "end": "13:00"},
                "b": {"start": "10:00", "end": "15:00"},
            },
            overlap_policy="last_wins",
        )
        s = _make_scheduler(tl, now=datetime(2026, 6, 1, 11, 0))
        result = s._find_active_period("11:00", {"periods": ["a", "b"]})
        assert result == "b"

    def test_missing_period_in_timeline_skipped(self):
        """If a period key listed in day_plan isn't in timeline.periods, skip it."""
        tl = _minimal_timeline(
            periods={"morning": {"start": "08:00", "end": "12:00"}},
            day_plans={"workday": {"periods": ["missing_key", "morning"]}},
        )
        # Skip validation to test runtime behavior
        s = _make_scheduler(tl, enabled=False)
        s.enabled = True
        s.timeline = tl
        result = s._find_active_period("10:00", {"periods": ["missing_key", "morning"]})
        assert result == "morning"


# ===================================================================
# _merge_with_default
# ===================================================================

class TestMergeWithDefault:

    def test_no_period_returns_default(self):
        default = {"collect": True, "analyze": False, "push": False,
                    "report_mode": "current", "ai_mode": "follow_report"}
        tl = _minimal_timeline(default=default)
        s = _make_scheduler(tl)
        merged = s._merge_with_default(None)
        assert merged["collect"] is True
        assert merged["analyze"] is False

    def test_period_overrides_default(self):
        default = {"collect": True, "analyze": False, "push": False,
                    "report_mode": "current", "ai_mode": "follow_report"}
        tl = _minimal_timeline(
            default=default,
            periods={"p1": {"start": "08:00", "end": "12:00",
                            "analyze": True, "push": True}},
        )
        s = _make_scheduler(tl)
        merged = s._merge_with_default("p1")
        assert merged["analyze"] is True
        assert merged["push"] is True
        assert merged["collect"] is True  # inherited from default

    def test_once_merged(self):
        default = {"collect": True, "analyze": False, "push": False,
                    "report_mode": "current", "ai_mode": "follow_report",
                    "once": {"analyze": False}}
        tl = _minimal_timeline(
            default=default,
            periods={"p1": {"start": "08:00", "end": "12:00",
                            "once": {"push": True}}},
        )
        s = _make_scheduler(tl)
        merged = s._merge_with_default("p1")
        assert merged["once"]["analyze"] is False  # from default
        assert merged["once"]["push"] is True       # from period


# ===================================================================
# _resolve_ai_mode
# ===================================================================

class TestResolveAiMode:

    def test_follow_report(self):
        assert Scheduler._resolve_ai_mode({"ai_mode": "follow_report", "report_mode": "daily"}) == "daily"

    def test_explicit_mode(self):
        assert Scheduler._resolve_ai_mode({"ai_mode": "weekly"}) == "weekly"

    def test_default(self):
        assert Scheduler._resolve_ai_mode({}) == "current"


# ===================================================================
# resolve()  – integration
# ===================================================================

class TestResolve:

    def test_disabled_returns_defaults(self):
        tl = _minimal_timeline()
        s = _make_scheduler(tl, enabled=False)
        r = s.resolve()
        assert r.day_plan == "disabled"
        assert r.collect is True
        assert r.analyze is True
        assert r.push is True

    def test_no_period_hit_uses_default(self):
        """At 03:00, no period matches -> default config."""
        tl = _minimal_timeline(
            periods={"morning": {"start": "08:00", "end": "12:00"}},
        )
        # 2026-06-01 is a Monday (isoweekday=1)
        s = _make_scheduler(tl, now=datetime(2026, 6, 1, 3, 0))
        r = s.resolve()
        assert r.period_key is None
        assert r.collect is True
        assert r.analyze is False

    def test_period_hit(self):
        tl = _minimal_timeline(
            periods={
                "morning": {"start": "08:00", "end": "12:00",
                            "name": "Morning", "analyze": True, "push": True,
                            "report_mode": "daily"},
            },
        )
        s = _make_scheduler(tl, now=datetime(2026, 6, 1, 10, 0))
        r = s.resolve()
        assert r.period_key == "morning"
        assert r.period_name == "Morning"
        assert r.analyze is True
        assert r.push is True
        assert r.report_mode == "daily"

    def test_cross_day_period_hit_at_night(self):
        tl = _minimal_timeline(
            periods={
                "night": {"start": "22:00", "end": "06:00",
                          "name": "Night Shift", "collect": True},
            },
        )
        s = _make_scheduler(tl, now=datetime(2026, 6, 1, 23, 30))
        r = s.resolve()
        assert r.period_key == "night"

    def test_cross_day_period_hit_early_morning(self):
        tl = _minimal_timeline(
            periods={
                "night": {"start": "22:00", "end": "06:00",
                          "name": "Night Shift", "collect": True},
            },
        )
        s = _make_scheduler(tl, now=datetime(2026, 6, 1, 4, 0))
        r = s.resolve()
        assert r.period_key == "night"

    def test_cross_day_period_excluded_at_end(self):
        tl = _minimal_timeline(
            periods={
                "night": {"start": "22:00", "end": "06:00",
                          "name": "Night Shift", "collect": True},
            },
        )
        s = _make_scheduler(tl, now=datetime(2026, 6, 1, 6, 0))
        r = s.resolve()
        assert r.period_key is None

    def test_week_map_routes_to_correct_day_plan(self):
        """Different day plans for different weekdays."""
        tl = _minimal_timeline(
            periods={
                "work": {"start": "09:00", "end": "17:00", "analyze": True},
                "rest": {"start": "09:00", "end": "17:00", "push": True},
            },
            day_plans={
                "workday": {"periods": ["work"]},
                "weekend": {"periods": ["rest"]},
            },
            week_map={
                1: "workday", 2: "workday", 3: "workday",
                4: "workday", 5: "workday",
                6: "weekend", 7: "weekend",
            },
        )
        # Monday 10:00 -> workday -> work period
        s = _make_scheduler(tl, now=datetime(2026, 6, 1, 10, 0))  # Mon
        r = s.resolve()
        assert r.period_key == "work"
        assert r.analyze is True

    def test_once_flags_propagated(self):
        tl = _minimal_timeline(
            default={"collect": True, "analyze": False, "push": False,
                      "report_mode": "current", "ai_mode": "follow_report"},
            periods={
                "p1": {"start": "08:00", "end": "12:00",
                       "analyze": True, "once": {"analyze": True, "push": True}},
            },
        )
        s = _make_scheduler(tl, now=datetime(2026, 6, 1, 10, 0))
        r = s.resolve()
        assert r.once_analyze is True
        assert r.once_push is True


# ===================================================================
# already_executed / record_execution
# ===================================================================

class TestExecutionTracking:

    def test_already_executed_delegates_to_storage(self):
        tl = _minimal_timeline()
        s = _make_scheduler(tl)
        s.storage.has_period_executed.return_value = True
        assert s.already_executed("p1", "analyze", "2026-06-01") is True
        s.storage.has_period_executed.assert_called_once_with("2026-06-01", "p1", "analyze")

    def test_record_execution_delegates_to_storage(self):
        tl = _minimal_timeline()
        s = _make_scheduler(tl)
        s.record_execution("p1", "push", "2026-06-01")
        s.storage.record_period_execution.assert_called_once_with("2026-06-01", "p1", "push")


# ===================================================================
# _build_timeline
# ===================================================================

class TestBuildTimeline:

    def test_custom_preset(self):
        tl = _minimal_timeline()
        s = _make_scheduler(tl)
        assert s.timeline["default"] is not None

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="未知的预设模板"):
            Scheduler(
                schedule_config={"enabled": True, "preset": "nonexistent"},
                timeline_data={"presets": {}},
                storage_backend=MagicMock(),
                get_time_func=lambda: datetime.now(),
            )

    def test_periods_none_defaults_to_empty_dict(self):
        tl = _minimal_timeline()
        tl["periods"] = None
        # Build via the preset=custom path
        schedule_config = {"enabled": False, "preset": "custom"}
        timeline_data = {"custom": tl}
        s = Scheduler(
            schedule_config=schedule_config,
            timeline_data=timeline_data,
            storage_backend=MagicMock(),
            get_time_func=lambda: datetime.now(),
        )
        assert s.timeline["periods"] == {}


# ===================================================================
# Edge cases specific to the half-open refactor
# ===================================================================

class TestHalfOpenEdgeCases:
    """Regression tests: behavior that changed or could break at boundaries."""

    def test_adjacent_three_periods(self):
        """Three adjacent periods: [08,12) [12,18) [18,22). No overlap."""
        tl = _minimal_timeline(
            periods={
                "a": {"start": "08:00", "end": "12:00"},
                "b": {"start": "12:00", "end": "18:00"},
                "c": {"start": "18:00", "end": "22:00"},
            },
            overlap_policy="error_on_overlap",
        )
        _make_scheduler(tl)  # validation must pass

    def test_full_day_coverage_no_overlap(self):
        """Four periods covering 00:00-24:00 with no overlap."""
        tl = _minimal_timeline(
            periods={
                "night": {"start": "00:00", "end": "06:00"},
                "morning": {"start": "06:00", "end": "12:00"},
                "afternoon": {"start": "12:00", "end": "18:00"},
                "evening": {"start": "18:00", "end": "00:00"},
            },
            overlap_policy="error_on_overlap",
        )
        _make_scheduler(tl)  # should not raise

    def test_full_day_resolve_boundary_midnight(self):
        """At 00:00, only the 'night' period [00:00,06:00) is active, not 'evening'."""
        tl = _minimal_timeline(
            periods={
                "night": {"start": "00:00", "end": "06:00", "name": "Night"},
                "evening": {"start": "18:00", "end": "00:00", "name": "Evening"},
            },
        )
        s = _make_scheduler(tl, now=datetime(2026, 6, 1, 0, 0))
        r = s.resolve()
        assert r.period_key == "night"

    def test_cross_day_and_normal_adjacent_at_end(self):
        """Cross-day [22:00,07:00) adjacent to normal [07:00,12:00). No overlap."""
        assert Scheduler._ranges_overlap("22:00", "07:00", "07:00", "12:00") is False

    def test_minute_before_end_is_included(self):
        """One minute before the end boundary is still inside the range."""
        assert Scheduler._in_range("11:59", "08:00", "12:00") is True

    def test_ranges_overlap_symmetric(self):
        """Overlap check must be symmetric."""
        assert Scheduler._ranges_overlap("08:00", "10:00", "09:00", "11:00") is True
        assert Scheduler._ranges_overlap("09:00", "11:00", "08:00", "10:00") is True

    def test_ranges_no_overlap_symmetric(self):
        assert Scheduler._ranges_overlap("08:00", "09:00", "09:00", "10:00") is False
        assert Scheduler._ranges_overlap("09:00", "10:00", "08:00", "09:00") is False
