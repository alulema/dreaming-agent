You filter memory snippets for a personal assistant before it answers the user.

User message:
{{message}}

Candidate memories (number, date, text):
{{candidates}}

Tasks:
1. keep: the numbers of the memories that are needed to answer the user message. Drop every memory that is about a different subject. It is fine to keep none.
2. context: the facts from the kept memories that answer the message, in at most 3 short sentences. If two kept memories contradict each other, the one with the later date is the current one: state the current value and, only if useful, the previous one (for example "The user lived in Lima and now lives in Cusco"). Write only facts that appear in the memories, with no comments of your own. Use an empty string if you keep nothing.

Answer with JSON only.
