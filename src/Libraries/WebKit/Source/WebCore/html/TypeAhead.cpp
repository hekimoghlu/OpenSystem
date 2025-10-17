/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 11, 2024.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#include "config.h"
#include "TypeAhead.h"

#include "KeyboardEvent.h"
#include <wtf/text/StringToIntegerConversion.h>
#include <wtf/unicode/CharacterNames.h>

namespace WebCore {

TypeAhead::TypeAhead(TypeAheadDataSource* dataSource)
    : m_dataSource(dataSource)
    , m_repeatingChar(0)
{
}

static const Seconds typeAheadTimeout { 1_s };

static String stripLeadingWhiteSpace(const String& string)
{
    unsigned length = string.length();
    unsigned i;
    for (i = 0; i < length; ++i) {
        if (string[i] != noBreakSpace && !deprecatedIsSpaceOrNewline(string[i]))
            break;
    }
    return string.substring(i, length - i);
}

int TypeAhead::handleEvent(KeyboardEvent* event, MatchModeFlags matchMode)
{
    if (event->timeStamp() < m_lastTypeTime)
        return -1;

    int optionCount = m_dataSource->optionCount();
    Seconds delta = event->timeStamp() - m_lastTypeTime;
    m_lastTypeTime = event->timeStamp();

    UChar c = event->charCode();

    if (delta > typeAheadTimeout)
        m_buffer.clear();
    m_buffer.append(c);

    if (optionCount < 1)
        return -1;

    int searchStartOffset = 1;
    String prefix;
    if (matchMode & CycleFirstChar && c == m_repeatingChar) {
        // The user is likely trying to cycle through all the items starting
        // with this character, so just search on the character.
        prefix = span(c);
        m_repeatingChar = c;
    } else if (matchMode & MatchPrefix) {
        prefix = m_buffer.toString();
        if (m_buffer.length() > 1) {
            m_repeatingChar = 0;
            searchStartOffset = 0;
        } else
            m_repeatingChar = c;
    }

    if (!prefix.isEmpty()) {
        int selected = m_dataSource->indexOfSelectedOption();
        int index = (selected < 0 ? 0 : selected) + searchStartOffset;
        index %= optionCount;

        String prefixWithCaseFolded(prefix.foldCase());
        for (int i = 0; i < optionCount; ++i, index = (index + 1) % optionCount) {
            // Fold the option string and check if its prefix is equal to the folded prefix.
            String text = m_dataSource->optionAtIndex(index);
            if (stripLeadingWhiteSpace(text).foldCase().startsWith(prefixWithCaseFolded))
                return index;
        }
    }

    if (matchMode & MatchIndex) {
        int index = parseIntegerAllowingTrailingJunk<int>(m_buffer).value_or(0);
        if (index > 0 && index <= optionCount)
            return index - 1;
    }
    return -1;
}

} // namespace WebCore
