"""Tests for the rule-based reminder extractor (scripts/semantic/reminder_rules.py). Pure functions, no I/O.
Run: .venv/bin/python scripts/tests/test_reminder_rules.py"""
import datetime as dt
import os
import sys
import unittest
from zoneinfo import ZoneInfo

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.append(os.path.join(REPO_ROOT, "scripts", "semantic"))

import reminder_rules as rr

IST = ZoneInfo("Asia/Kolkata")
SUNDAY = "2026-09-20 10:00"     # a Sunday morning in IST


def at(when):
    return int(dt.datetime.fromisoformat(when).replace(tzinfo=IST).timestamp())


def one(text, when=SUNDAY, **kw):
    found = rr.extract(text, at(when), **kw)
    return found[0] if found else None


def local_time(c):
    return dt.datetime.fromtimestamp(c.due_at, IST).strftime("%H:%M")


class DatesAndTimes(unittest.TestCase):
    def test_hinglish_kal_with_time_is_tomorrow_evening(self):
        c = one("kal 6 baje call karenge")
        self.assertEqual((c.date_local, local_time(c), c.all_day, c.kind), ("2026-09-21", "18:00", False, "meeting"))

    def test_kal_in_the_past_tense_is_yesterday_and_skipped(self):
        self.assertIsNone(one("kal market gaya tha"))

    def test_iso_date_with_ist_time(self):
        c = one("Hi Aarav, The discussion on 2026-01-19 at 10:00 IST will be a video call.", "2026-01-17 09:00")
        self.assertEqual((c.date_local, local_time(c)), ("2026-01-19", "10:00"))
        self.assertTrue(c.title.startswith("The discussion"))    # greeting stripped

    def test_day_month_resolves_against_the_message_date(self):
        c = one("Maa's knee surgery is scheduled for 12 March at Pink Cedar Hospital.", "2025-03-04 09:00")
        self.assertEqual((c.date_local, c.kind, c.all_day), ("2025-03-12", "appointment", True))

    def test_day_month_early_next_year(self):
        self.assertEqual(one("exam on 5 Jan, padhai shuru karni hai", "2026-12-10 09:00").date_local, "2027-01-05")

    def test_slash_dates_are_day_first(self):
        self.assertEqual(one("exam on 12/10").date_local, "2026-10-12")

    def test_date_nearest_the_reason_wins(self):
        c = one("I will be in Jaipur from 9–19 March for my mother's knee surgery on 12 March", "2025-03-05 09:00")
        self.assertEqual(c.date_local, "2025-03-12")

    def test_weekdays(self):
        self.assertEqual(one("agle monday milte hai").date_local, "2026-09-28")
        self.assertEqual(one("Friday tak submit kar dena").date_local, "2026-09-25")
        self.assertEqual(one("Sat ko plan hai?").date_local, "2026-09-26")
        self.assertEqual(one("meet on Sunday at 5pm").date_local, "2026-09-27")

    def test_parso_and_part_of_day(self):
        self.assertEqual(one("parso movie chalenge?").date_local, "2026-09-22")
        c = one("bring the notes to the library steps tomorrow morning")
        self.assertEqual((c.date_local, local_time(c)), ("2026-09-21", "09:00"))

    def test_bare_time_reminder_is_today_or_tomorrow(self):
        self.assertEqual(one("remind me to call papa at 7pm").date_local, "2026-09-20")
        self.assertEqual(one("remind me to call papa at 7am").date_local, "2026-09-21")    # 7am has passed


class WhatIsNotAReminder(unittest.TestCase):
    def test_chatter_and_negations(self):
        for text in ["going for filter coffee now", "not posting the final shortlist tonight",
                     "No need to reopen icon row in the orange draft tonight; just save the sketch.",
                     "exactly, tomorrow you might have a clearer sentence about the doorway",
                     "thinking of making khichdi at home tomorrow morning",
                     "We missed the train at Yeshwanthpur. I read the departure as 22:10; it was 21:10.",
                     "Hi Aarav, I reviewed the query example for 2026-08-18 and the failure case is clearer.",
                     "Back at the Friday idea, trying to understand why the showtime clashes",
                     "Can we commit related changes together in the morning debug session?",
                     "Searched for Bengaluru Jaipur flight dates tomorrow"]:
            self.assertIsNone(one(text), text)

    def test_hinglish_stories_about_the_past(self):
        for text in ["maine kal 5 baje chai piya tha", "kal raat 3 baje soyi thi", "aaj maine movie dekhi",
                     "aaj dinner mein sabzi bana rha", "parso uski 6 baje tak class thi", "Then 2 baje lunch hua",
                     "Date: Sat Sep 5 13:53:01 2026 +0530"]:
            self.assertIsNone(one(text), text)

    def test_hinglish_plans(self):
        self.assertEqual(one("parso 2 exam hai").date_local, "2026-09-22")
        self.assertEqual(one("kal 7:30 library jana hai").date_local, "2026-09-21")
        self.assertEqual(one("kal subah gym jaunga").date_local, "2026-09-21")
        self.assertEqual(one("parso hackathon ka deadline h").kind, "deadline")

    def test_bhool_mat_is_not_a_negation(self):
        self.assertEqual(one("assignment submit karna hai friday tak, bhool mat").kind, "deadline")


class Birthdays(unittest.TestCase):
    def test_wish_becomes_a_yearly_reminder(self):
        c = one("Happy 21st birthday, Aarav! Call home when your classes finish.", "2026-04-07 08:00", birthday_of="Aarav")
        self.assertEqual((c.title, c.kind, c.repeat, c.date_local), ("Aarav's birthday", "birthday", "yearly", "2027-04-07"))

    def test_wish_without_a_person_is_ignored(self):
        self.assertIsNone(one("Happy birthday!!", "2026-04-07 08:00"))


if __name__ == "__main__":
    unittest.main()
