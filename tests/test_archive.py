"""Offline tests for archive.py (no servers needed). Run: python3 -m unittest discover tests"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import archive  # noqa: E402

LIMIT = 20000


def chat(turns, filler=3000):
    msgs = [{"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "I'll share notes over this chat. Keep track of them."},
            {"role": "assistant", "content": "Go ahead."}]
    for i in range(turns):
        msgs.append({"role": "user", "content": f"Note {i}: the code word for note {i} is word{i}.\n\n" + "lorem ipsum " * (filler // 12)})
        msgs.append({"role": "assistant", "content": f"Noted: note {i}."})
    return msgs


class ArchiveTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.a = archive.Archive(os.path.join(self.dir.name, "t.sqlite"))

    def tearDown(self):
        self.a.db.close()
        self.dir.cleanup()

    def prepare(self, msgs):
        payload = {"messages": [dict(m) for m in msgs]}
        return payload, self.a.prepare(payload, "tester", LIMIT)

    def test_short_chat_untouched(self):
        payload, prep = self.prepare(chat(3))
        self.assertIsNone(prep)
        self.assertEqual(payload["messages"], chat(3))

    def test_long_chat_is_cut_and_first_message_pinned(self):
        msgs = chat(40)
        payload, (conv, query, info) = self.prepare(msgs)
        out = payload["messages"]
        self.assertTrue(info["new_cut"])
        self.assertEqual(out[0], msgs[0])
        self.assertEqual(out[1], msgs[1])  # first user message kept
        self.assertEqual(out[-1], msgs[-1])
        kept = sum(archive.tokens(m) for m in out)
        self.assertLess(kept, LIMIT * 0.6)  # archived down to about half the limit
        self.assertIn(out[2]["role"], ("user", "assistant"))  # never starts the kept part with a tool result

    def test_prefix_stable_until_next_cut(self):
        msgs = chat(40)
        p1, (conv, _, info1) = self.prepare(msgs)
        self.a.add_note(p1, conv, info1, False)
        msgs2 = msgs + [{"role": "user", "content": "short question"}, {"role": "assistant", "content": "short answer"}]
        p2, (_, _, info2) = self.prepare(msgs2)
        self.a.add_note(p2, conv, info2, False)
        self.assertFalse(info2["new_cut"])
        self.assertEqual(p2["messages"][:len(p1["messages"])], p1["messages"])

    def request(self, msgs):
        """prepare + recall + note, as the gate does for one request."""
        p, (conv, _, info) = self.prepare(msgs)
        q = self.a.recall_query(p, conv, True)
        n = self.a.recall(p, conv, q, None, True)
        self.a.add_note(p, conv, info, True)
        return p, n, q

    def test_agent_tool_loop_keeps_prefix(self):
        msgs = chat(40) + [{"role": "user", "content": "What was the code word for note 3?"}]
        p1, n1, q1 = self.request(msgs)
        self.assertTrue(q1 and n1 > 0)
        for step in range(3):  # tool rounds: same user message, new tool results at the end
            msgs = msgs + [{"role": "assistant", "content": "", "tool_calls": [
                               {"id": f"c{step}", "type": "function", "function": {"name": "read", "arguments": "{}"}}]},
                           {"role": "tool", "tool_call_id": f"c{step}", "content": f"result {step}"}]
            p2, n2, q2 = self.request(msgs)
            self.assertIsNone(q2)  # excerpts reused, no new search
            self.assertEqual(n2, n1)
            self.assertEqual(p2["messages"][:len(p1["messages"])], p1["messages"])
            p1 = p2
        p3, _, q3 = self.request(msgs + [{"role": "user", "content": "And note 5?"}])
        self.assertTrue(q3)  # a new user message gets its own excerpts
        self.assertNotIn("Archived excerpts", str(p3["messages"][-2]))

    def test_index_read_and_search(self):
        payload, (conv, _, info) = self.prepare(chat(40))
        self.a.add_note(payload, conv, info, True)
        system = payload["messages"][0]["content"]
        self.assertIn("Index of the archived messages", system)
        self.assertIn("archive_read", system)
        self.assertIn("“Note 0: the code word", system)  # quoted openings
        text = self.a.read(conv, 3, 3)  # message 3 = first note (message 1 is the pinned first user message)
        self.assertIn("word0", text)
        self.assertIn("not archived", self.a.read(conv, info["cut"] + 5, info["cut"] + 6))
        self.assertIn("word7", self.a.search_text(conv, "code word for note 7", None))

    def test_summaries_replace_nothing_but_add(self):
        payload, (conv, _, info) = self.prepare(chat(40))
        pending = self.a.unsummarized(conv)
        self.assertTrue(pending)
        self.a.set_summaries(conv, {pending[0][0]: "Notes 0-1 with code words"})
        self.a.add_note(payload, conv, info, False)
        self.assertIn("Notes 0-1 with code words — user: “Note 0", payload["messages"][0]["content"])
        self.assertEqual(len(self.a.unsummarized(conv)), len(pending) - 1)

    def test_edited_history_starts_over(self):
        msgs = chat(40)
        _, (conv, _, _) = self.prepare(msgs)
        edited = [dict(m) for m in msgs]
        edited[5]["content"] = "something else entirely"
        _, (conv2, _, info) = self.prepare(edited)
        self.assertEqual(conv, conv2)
        self.assertIn("something else", self.a.read(conv, 5, 5))

    def test_delete_conv_clears_search(self):
        _, (conv, _, _) = self.prepare(chat(40))
        archive.delete_conv(self.a.db, conv)
        self.assertEqual(self.a.db.execute("select count(*) from chunks").fetchone()[0], 0)
        hits = self.a.db.execute("select count(*) from chunks_fts where chunks_fts match 'word3'").fetchone()[0]
        self.assertEqual(hits, 0)

    def test_tool_results_stay_with_their_call(self):
        msgs = chat(2)
        for i in range(30):
            msgs.append({"role": "user", "content": f"step {i}"})
            msgs.append({"role": "assistant", "content": "", "tool_calls": [
                {"id": f"c{i}", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]})
            msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "x " * 1500})
            msgs.append({"role": "assistant", "content": f"done {i}"})
        payload, prep = self.prepare(msgs)
        self.assertIsNotNone(prep)
        out = payload["messages"]
        for i, m in enumerate(out):
            if m["role"] == "tool":
                self.assertTrue(out[i - 1].get("tool_calls") or out[i - 1]["role"] == "tool")


if __name__ == "__main__":
    unittest.main()
