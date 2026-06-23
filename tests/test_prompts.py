import unittest

from moondream_mini.prompts import build_prompt, extract_answer, resolve_label_space


class PromptTests(unittest.TestCase):
    def test_infers_count_prompt(self):
        task_type, label_space = resolve_label_space(None, None, "How many workers are visible?")
        self.assertEqual((task_type, label_space), ("count", "count_4"))
        self.assertIn("Options: 0, 1, 2, 3+", build_prompt("How many workers are visible?", label_space))

    def test_extracts_constrained_answers(self):
        self.assertEqual(extract_answer("Answer: 3 </s>", "count_4"), "3+")
        self.assertEqual(extract_answer("Answer: center worker", "location_3"), "center")
        self.assertEqual(extract_answer("Answer: yes </s>", "yes_no"), "yes")


if __name__ == "__main__":
    unittest.main()
