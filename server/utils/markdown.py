# -*- coding: utf-8 -*-
import re
import os
import codecs
import json
from typing import Text, Optional, Tuple, Match
from warnings import simplefilter
from utils import log_util

# ignore all warnings

simplefilter(action='ignore')

##Global parameters
scriptDir = os.path.dirname(__file__)
datapath = os.path.join(scriptDir, '..', 'training_data', 'intents')

INTENT = "intent"
SYNONYM = "synonym"
REGEX = "regex"
LOOKUP = "lookup"

available_sections = [INTENT, SYNONYM, REGEX, LOOKUP]
current_section = None
current_title = None

item_regex = re.compile(r"\s*[-*+]\s*(.+)")
comment_regex = re.compile(r"<!--[\s\S]*?--!*>", re.MULTILINE)
fname_regex = re.compile(r"\s*([^-*+]+)")
entity_regex = re.compile(
    r"\[(?P<entity_text>[^\]]+?)\](\((?P<entity>[^:)]+?)(?:\:(?P<value>[^)]+))?\)|\{(?P<entity_dict>[^}]+?)\})"
)

ESCAPE_DCT = {"\b": "\\b", "\f": "\\f", "\n": "\\n", "\r": "\\r", "\t": "\\t"}
ESCAPE = re.compile(r"[\b\f\n\r\t]")

GROUP_COMPLETE_MATCH = 0
current_section = ""
current_intent = ""
intent = []
utterance = []


def encode_string(s: Text) -> Text:
    """Return a encoded python string."""

    def replace(match: Match) -> Text:
        return ESCAPE_DCT[match.group(GROUP_COMPLETE_MATCH)]

    return ESCAPE.sub(replace, s)


def find_section_header(line: Text) -> Optional[Tuple[Text, Text]]:
    """Checks if the current line contains a section header
    and returns the section and the title."""
    match = re.search(r"##\s*(.+?):(.+)", line)
    if match is not None:
        return match.group(1), match.group(2)
    return None


def set_current_section(section: Text, title: Text) -> None:
    """Update parsing mode."""
    if section not in available_sections:
        log_util.log_errormsg(
            f"[MARKDOWN] found markdown section {section} which is not in the allowed sections {', '.join(available_sections)}.")
        raise ValueError(
            f"[MARKDOWN] found markdown section {section} which is not in the allowed sections {', '.join(available_sections)}.")

    global current_section, current_intent
    current_section = section
    current_intent = title


def parse_item(line: Text) -> None:
    """Parses an md list item line based on the current section type."""
    match = re.match(item_regex, line)
    if match:
        item = match.group(1)
        plain_text = re.sub(
            entity_regex, lambda m: m.groupdict()["entity_text"], item
        )
        if current_section == INTENT:
            utterance.append(plain_text)
            intent.append(current_intent)
        else:
            pass
    else:
        pass


def process_data(domain: Text, locale: Text) -> None:
    global utterance
    global intent
    #clear the list before loading
    utterance.clear()
    intent.clear()
    try:
        file = codecs.open(os.path.join(datapath, domain + '_' + locale + ".md"), 'r', 'utf-8')
        lines = file.read().split("\n")
        log_util.log_infomsg(f"[MARKDOWN] received data, total lines: {len(lines)}")
        for line in lines:
            line = line.strip()
            header = find_section_header(line)
            if header:
                set_current_section(header[0], header[1])
            else:
                parse_item(line)
        return utterance, intent
    except FileNotFoundError:
        log_util.log_errormsg(
            f"[MARKDOWN] missing file for domain {domain}, ensure file is present in given .md format.")
        return utterance, intent
