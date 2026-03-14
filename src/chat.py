import re
from huggingface_hub import InferenceClient
from config import BASE_MODEL, MY_MODEL, HF_TOKEN
from retrieval.search import search_courses

# Maps MIT course number shorthand → department name substring used in the catalog
COURSE_NUMBER_TO_DEPT = {
    "1":  "Civil and Environmental",
    "2":  "Mechanical Engineering",
    "3":  "Materials Science",
    "4":  "Architecture",
    "5":  "Chemistry",
    "6":  "Electrical Engineering and Computer Science",
    "7":  "Biology",
    "8":  "Physics",
    "9":  "Brain and Cognitive",
    "10": "Chemical Engineering",
    "11": "Urban Studies",
    "12": "Earth, Atmospheric",
    "14": "Economics",
    "15": "Management",
    "16": "Aeronautics",
    "17": "Political Science",
    "18": "Mathematics",
    "20": "Biological Engineering",
    "22": "Nuclear Science",
    "24": "Linguistics and Philosophy",
}

# Only values that actually exist in courses.json distribution_requirements:
# CI-HW (19), HASS-A (172), HASS-H (394), HASS-S (238), REST (67), Institute Lab (26)
# Note: "HASS" and "HASS-E" and "CI-H" do NOT exist as standalone values.
DIST_ALIASES = {
    "ci-hw": "CI-HW",
    "ci-h":  "CI-HW",   # CI-H is fulfilled by CI-HW in the catalog
    "hass-a": "HASS-A",
    "hass-h": "HASS-H",
    "hass-s": "HASS-S",
    "rest":  "REST",
}

# Generic "HASS" mention — no single filter value covers all HASS subtypes,
# so we run three searches and merge the results.
HASS_SUBTYPES = ["HASS-A", "HASS-H", "HASS-S"]


# Common department aliases students use that aren't "Course N" or "N-X"
DEPT_ALIASES = {
    r"\b(cs|eecs|computer\s+science)\b": "Electrical Engineering and Computer Science",
    r"\b(math|mathematics)\b":           "Mathematics",
    r"\bphysics\b":                       "Physics",
    r"\b(bio\b|biology)\b":               "Biology",
    r"\b(econ|economics)\b":              "Economics",
    r"\baero(nautics)?\b":                "Aeronautics",
    r"\b(nuke|nuclear)\b":                "Nuclear Science",
    r"\barchitecture\b":                  "Architecture",
    r"\b(mech\s+e|mechanical\s+engineering)\b": "Mechanical Engineering",
}


# Referential language that signals a follow-up about previously discussed courses
_REFERENTIAL = re.compile(
    r"\b(these|those|them|they|it|the course|the class|the ones?|that course|that class"
    r"|above|mentioned|recommended|you said|you mentioned|earlier|just said)\b",
    re.I,
)


def _build_search_query(user_input: str, recent_user_text: str) -> str:
    """
    For short or referential follow-up messages, augment the search query with
    recent history so retrieval stays grounded in the ongoing conversation.
    """
    is_followup = len(user_input.split()) < 8 or bool(_REFERENTIAL.search(user_input))
    if is_followup and recent_user_text:
        return f"{recent_user_text[-300:]} {user_input}".strip()
    return user_input


def _extract_filters(text: str) -> dict:
    """
    Extract structured search filters from natural language.
    Returns 'departments' as a list to support multi-department queries.
    """
    filters = {}
    lower = text.lower()
    departments = []

    # Collect ALL "Course N" mentions (supports multi-department queries)
    for m in re.finditer(r"\bcourse\s+(\d+)\b", lower):
        num = m.group(1)
        dept = COURSE_NUMBER_TO_DEPT.get(num)
        if dept and dept not in departments:
            departments.append(dept)

    # If no "Course N" found, try "N-X" format (e.g. 6-3, 18-c)
    if not departments:
        for m in re.finditer(r"\b(\d+)-[1-4p]\b", lower):
            num = m.group(1)
            dept = COURSE_NUMBER_TO_DEPT.get(num)
            if dept and dept not in departments:
                departments.append(dept)

    # Common aliases (only if no Course-N match already)
    if not departments:
        for pattern, dept in DEPT_ALIASES.items():
            if re.search(pattern, lower):
                departments.append(dept)
                break  # one alias match per query to avoid false positives

    if departments:
        filters["departments"] = departments

    # Distribution requirement — check specific subtypes first, then generic HASS
    for alias, canonical in DIST_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", lower):
            filters["distribution_requirement"] = canonical
            break
    else:
        # Generic "hass" with no subtype specified
        if re.search(r"\bhass\b", lower):
            filters["distribution_requirement"] = "HASS_ANY"

    # Prereqs flag: "no prereqs", "no prerequisites"
    if re.search(r"\bno\s+pre-?req", lower):
        filters["has_prereqs"] = False

    return filters


SYSTEM_PROMPT = """You are a helpful MIT course catalog assistant. You help MIT students find courses that match their academic requirements, interests, and scheduling constraints.

STRICT GROUNDING RULE: You will be given course data retrieved from the MIT catalog. You may ONLY recommend courses that appear word-for-word in that data. Never invent, guess, or recall course numbers, titles, or details from your own training. If the data does not contain a suitable course, say so clearly — do not fill the gap with made-up suggestions.

Behavior guidelines:
- If the student's request is vague, ask 1-3 focused clarifying questions before recommending. Useful things to ask about: distribution requirements still needed (CI-H, HASS, REST, etc.), topics of interest, year and major, scheduling preferences (morning vs afternoon, specific days free).
- When recommending courses, present 3-5 options from the provided data. For each, include: course number and title, prerequisites, distribution requirements fulfilled, and a brief explanation of why it fits.
- Be honest when information is missing. If schedule or instructor data is absent, say so and direct the student to student.mit.edu/catalog for current semester details.
- A note on requirements: CI-HW courses satisfy the CI-H requirement (they just have a writing component). Flag this if relevant.
- When a student asks generically for "HASS courses" with no other constraints, don't try to find a perfect match — just pick 2-3 examples from the data (spanning different HASS subtypes if possible), briefly describe them, and note that there are many HASS options and you can help narrow down if they share more about their interests.
- Keep responses focused and conversational. Don't list every field for every course — highlight what's relevant.
- When a student asks a follow-up question about courses you already recommended (e.g. "when do these meet?", "what are the prereqs for those?"), stay focused on those specific courses. If schedule or other detail is missing from the provided data, say so clearly and direct them to student.mit.edu/catalog — do not switch to discussing different courses.
- If the student asks for courses from a specific department or field and the provided data doesn't contain clear matches, say so and ask them to rephrase using MIT's course number system (e.g. "Course 6 for CS/EECS", "Course 5 for Chemistry", "Course 18 for Math"). Do not guess or invent courses to fill the gap.
- If the student requests courses from multiple departments in one message, address each department separately in your response.

Handling inherited filters:
- If the context begins with a "[Filter applied from earlier conversation: ...]" note, acknowledge it naturally — e.g. "Since you mentioned CI-H earlier, I've filtered for that — let me know if you'd like a broader search." Do not ignore it silently.
- If the student's current question seems unrelated to that requirement (e.g. they're now asking about a specific topic with no mention of requirements), gently ask whether they still want it applied before presenting results.

Handling partial and missing matches:
- Always recommend the most relevant courses from the provided data, even if they don't satisfy every constraint.
- If no course satisfies all of the student's constraints (e.g. right topic AND right distribution requirement), say so in one sentence upfront, then still present the closest available options and note which constraint each one is missing.
- If the data contains no relevant courses at all, say so honestly and suggest the student check student.mit.edu/catalog directly."""


class Chatbot:
    """
    This class is extra scaffolding around a model. Modify this class to specify how the model recieves prompts and generates responses.

    Example usage:
        chatbot = Chatbot()
        response = chatbot.get_response("What options are available for me?")
    """

    def __init__(self):
        """
        Initialize the chatbot with a HF model ID
        """
        model_id = MY_MODEL if MY_MODEL else BASE_MODEL
        self.client = InferenceClient(model=model_id, token=HF_TOKEN)
        self.last_courses: list = []  # courses retrieved in the most recent non-follow-up turn

    def _format_course(self, course: dict) -> str:
        """Format a single course dict into a compact, readable text block."""
        lines = []

        num = course.get("course_number", "")
        title = course.get("title", "")
        lines.append(f"[{num}] {title}")

        depts = course.get("departments") or []
        if depts:
            lines.append(f"  Department: {', '.join(depts)}")

        dist = course.get("distribution_requirements") or []
        if dist:
            lines.append(f"  Distribution: {', '.join(dist)}")

        level = course.get("level") or []
        semesters = course.get("semesters_offered") or []
        meta = []
        if level:
            meta.append(", ".join(level))
        if semesters:
            meta.append(f"offered {', '.join(semesters)}")
        if meta:
            lines.append(f"  {' | '.join(meta)}")

        prereqs = course.get("prerequisites")
        lines.append(f"  Prerequisites: {prereqs if prereqs else 'None'}")

        units = course.get("units")
        if units:
            lines.append(f"  Units: {units}")

        schedule = course.get("schedule") or {}
        if schedule:
            sched_str = "; ".join(f"{k}: {v}" for k, v in schedule.items())
            lines.append(f"  Schedule: {sched_str}")
        elif course.get("schedule_notes"):
            lines.append(f"  Schedule: {course['schedule_notes']}")

        desc = course.get("description")
        if desc:
            lines.append(f"  Description: {desc[:300]}{'...' if len(desc) > 300 else ''}")

        return "\n".join(lines)

    def _parse_clean_history(self, history) -> list:
        """
        Parse Gradio history into clean [{role, content}] dicts.
        Fix #2: strips injected RAG context (everything after \\n\\n---\\n)
        from prior user messages so history doesn't bloat over turns.
        """
        clean = []
        if not history:
            return clean
        for msg in history:
            try:
                if isinstance(msg, dict):
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        content = " ".join(
                            p.get("text", "") for p in content
                            if isinstance(p, dict) and "text" in p
                        )
                    content = str(content)
                    # Strip RAG context block from prior user turns
                    if role == "user" and "\n\n---\n" in content:
                        content = content.split("\n\n---\n")[0]
                    clean.append({"role": role, "content": content})
                elif isinstance(msg, (list, tuple)):
                    user_msg = str(msg[0]) if len(msg) > 0 else ""
                    assistant_msg = str(msg[1]) if len(msg) > 1 else ""
                    if "\n\n---\n" in user_msg:
                        user_msg = user_msg.split("\n\n---\n")[0]
                    if user_msg:
                        clean.append({"role": "user", "content": user_msg})
                    if assistant_msg:
                        clean.append({"role": "assistant", "content": assistant_msg})
            except Exception:
                pass
        return clean

    def format_prompt(self, user_input, history=None):
        """
        Retrieves relevant courses for the user's query, formats them as context,
        and builds the full message list to send to the LLM.
        """
        # Fix #2: parse history into clean messages (RAG context stripped)
        clean_history = self._parse_clean_history(history)

        # Fix #1: build retrieval context from recent conversation history
        # Use last 3 user turns to inherit department/topic signals
        recent_user_text = " ".join(
            m["content"] for m in clean_history[-6:] if m["role"] == "user"
        )

        # Extract filters from current message
        filters_now = _extract_filters(user_input)
        dist_req = filters_now.pop("distribution_requirement", None)
        departments = filters_now.pop("departments", [])
        has_prereqs = filters_now.pop("has_prereqs", None)

        # Inherit department/dist from history only if current message has none
        dist_req_inherited = False
        if recent_user_text and (not departments or dist_req is None):
            hist_filters = _extract_filters(recent_user_text)
            if not departments:
                departments = hist_filters.pop("departments", [])
            if dist_req is None:
                inherited = hist_filters.pop("distribution_requirement", None)
                if inherited:
                    dist_req = inherited
                    dist_req_inherited = True

        # Remaining scalar filters (has_prereqs, etc.)
        base_filters = {}
        if has_prereqs is not None:
            base_filters["has_prereqs"] = has_prereqs

        # Detect referential follow-up — student is asking about courses just discussed
        is_referential = bool(_REFERENTIAL.search(user_input))

        context_parts = []
        retrieved_this_turn = []  # accumulated for self.last_courses update

        if is_referential and self.last_courses:
            # Reuse exact courses from previous turn — no re-retrieval needed.
            # The LLM has the same data it used to make its recommendation,
            # so it can answer detail questions (schedule, prereqs, etc.) directly.
            blocks = [self._format_course(c) for c in self.last_courses]
            context_parts.append(
                "Courses from previous recommendation (student is asking a follow-up):\n\n"
                + "\n\n".join(blocks)
            )
            # Don't update last_courses — the set being discussed hasn't changed

        else:
            # Fresh search — build an augmented query for short messages
            search_query = _build_search_query(user_input, recent_user_text)

            # Build a clean semantic query by stripping noise words AND department
            # mentions (dept is already handled by the filter; leaving it in biases
            # the embedding toward generic dept results instead of the actual topic).
            topic_query = re.sub(
                r"\b(course\s+\d+|\d+-[1-4p]|hass(-[ahs])?|ci-?h(w)?|rest|courses?"
                r"|classes?|that (match|satisfy|fulfill)|requirement|elective"
                r"|give me|show me|find me|i need|i want)\b",
                " ", user_input, flags=re.I
            )
            topic_query = re.sub(r"\s{2,}", " ", topic_query).strip()
            has_topic = len(topic_query) > 3

            # Use the clean topic string for semantic search; fall back to the
            # history-augmented query only when there's no identifiable topic.
            semantic_query = topic_query if has_topic else search_query

            # Run separate topic search per department for multi-dept queries
            if has_topic:
                if departments:
                    for dept in departments:
                        dept_filters = {**base_filters, "department": dept}
                        results = search_courses(search_query, filters=dept_filters, top_k=8)
                        retrieved_this_turn.extend(results)
                        if results:
                            blocks = [self._format_course(c) for c in results]
                            context_parts.append(
                                f"Topic matches in {dept}:\n\n" + "\n\n".join(blocks)
                            )
                        else:
                            context_parts.append(f"Topic matches in {dept}: No courses found.")
                else:
                    results = search_courses(
                        search_query, filters=base_filters if base_filters else None, top_k=10
                    )
                    retrieved_this_turn.extend(results)
                    if results:
                        blocks = [self._format_course(c) for c in results]
                        context_parts.append(
                            "Topic matches (courses relevant to your query):\n\n" + "\n\n".join(blocks)
                        )
                    else:
                        context_parts.append("Topic matches: No courses found for this query.")

            # Distribution requirement search
            if dist_req == "HASS_ANY":
                seen = set()
                dist_results = []
                for subtype in HASS_SUBTYPES:
                    sub_filters = {**base_filters, "distribution_requirement": subtype}
                    if len(departments) == 1:
                        sub_filters["department"] = departments[0]
                    for c in search_courses(search_query, filters=sub_filters, top_k=4):
                        if c["course_number"] not in seen:
                            dist_results.append(c)
                            seen.add(c["course_number"])
                retrieved_this_turn.extend(dist_results)
                label = "HASS (any subtype) courses" if has_topic else "Courses matching HASS (any subtype)"
                if dist_results:
                    blocks = [self._format_course(c) for c in dist_results]
                    context_parts.append(f"{label}:\n\n" + "\n\n".join(blocks))
                else:
                    context_parts.append("No HASS courses found in the catalog.")
            elif dist_req:
                dist_filters = {**base_filters, "distribution_requirement": dist_req}
                if len(departments) == 1:
                    dist_filters["department"] = departments[0]
                dist_results = search_courses(search_query, filters=dist_filters, top_k=10)
                retrieved_this_turn.extend(dist_results)
                label = f"{dist_req} courses" if has_topic else f"Courses matching {dist_req}"
                if dist_results:
                    blocks = [self._format_course(c) for c in dist_results]
                    context_parts.append(f"{label}:\n\n" + "\n\n".join(blocks))
                else:
                    context_parts.append(f"No {dist_req} courses found in the catalog.")

            # Fallback: if neither search ran (no topic, no dist filter)
            if not context_parts:
                results = search_courses(search_query, top_k=10)
                retrieved_this_turn.extend(results)
                blocks = [self._format_course(c) for c in results]
                context_parts.append("Relevant courses:\n\n" + "\n\n".join(blocks))

            # Update cache with all courses retrieved this turn
            if retrieved_this_turn:
                self.last_courses = retrieved_this_turn

        # Prepend inherited-filter notice so the LLM can acknowledge it
        if dist_req_inherited:
            notice = f"[Filter applied from earlier conversation: {dist_req}]"
            context_parts.insert(0, notice)

        context = "\n\n---\n\n".join(context_parts)

        # Build the message list using clean (non-bloated) history
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in clean_history:
            messages.append(msg)

        # Inject retrieved context into the current user message
        augmented_input = f"{user_input}\n\n---\n{context}"
        messages.append({"role": "user", "content": augmented_input})

        return messages

    def get_response(self, user_input, history=None):
        """
        Formats the prompt with RAG context and returns the model's response.
        """
        messages = self.format_prompt(user_input, history)
        response = self.client.chat_completion(messages)
        return response.choices[0].message.content.strip()
