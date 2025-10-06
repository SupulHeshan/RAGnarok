import json, re
s='''```json
{
  "answer": "The additional features to be added in the **Snake game** assignment are:\n 1. **Interactive sound effects** using a buzzer [doc:1, p:1].\n 2. A **simple menu** at the start to either begin a new game or view the past **high score** (saved in **EEPROM**) [doc:1, p:1].\n 3. **Joystick-controlled menu navigation** [doc:1, p:1].",

  "sources": [
    {
      "doc_id": 1,
      "page": 1,
      "quote": "Additional Features: \n      o Interactive sound effects using a buzzer. \n      o A simple menu at the beginning to start a new game or view the past high score. \n      o The high score should be saved in EEPROM. \n      o The menu navigation will be controlled by the joystick."
    }
  ]
}
```'''

def extract_json_from_text(text: str):
    if not isinstance(text, str):
        raise ValueError("LLM returned non-string answer")

    s = text.strip()
    s = re.sub(r"^```(?:json|\w+)?\s*", "", s, flags=re.I)
    s = re.sub(r"\s*```$", "", s)

    try:
        return json.loads(s)
    except Exception:
        pass

    start = None
    for i, ch in enumerate(s):
        if ch in '{[':
            start = i
            break
    if start is None:
        raise ValueError('No JSON object found in LLM output')

    stack = []
    end = None
    pairs = {'{': '}', '[': ']'}
    open_ch = s[start]
    close_ch = pairs[open_ch]
    for i in range(start, len(s)):
        ch = s[i]
        if ch == open_ch:
            stack.append(ch)
        elif ch == close_ch:
            stack.pop()
            if not stack:
                end = i
                break

    if end is None:
        last = s.rfind(close_ch)
        if last == -1:
            raise ValueError('Could not find end of JSON in LLM output')
        end = last

    candidate = s[start:end+1]

    try:
        return json.loads(candidate)
    except Exception:
        def escape_newlines_in_strings(text: str) -> str:
            pattern = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"', re.DOTALL)
            def _repl(m):
                inner = m.group(1)
                inner_escaped = inner.replace('\r', '\\r').replace('\n', '\\n')
                return f'"{inner_escaped}"'
            return pattern.sub(_repl, text)

        try:
            candidate_esc = escape_newlines_in_strings(candidate)
            return json.loads(candidate_esc)
        except Exception:
            fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
            fixed = escape_newlines_in_strings(fixed)
            try:
                return json.loads(fixed)
            except Exception as e:
                m = re.search(r"(\{[\s\S]*\})", s)
                if m:
                    try:
                        cand2 = escape_newlines_in_strings(m.group(1))
                        return json.loads(cand2)
                    except Exception:
                        pass
                raise e

# run the test
try:
    parsed = extract_json_from_text(s)
    print('PARSED OK')
    print('keys:', list(parsed.keys()))
    print('answer snippet:', parsed['answer'][:80])
    print('sources count:', len(parsed['sources']))
except Exception as e:
    print('PARSE FAILED', e)
