/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 17, 2024.
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
#include "BufferedLineReader.h"

#include <wtf/unicode/CharacterNames.h>

namespace WebCore {

std::optional<String> BufferedLineReader::nextLine()
{
    if (m_maybeSkipLF) {
        // We ran out of data after a CR (U+000D), which means that we may be
        // in the middle of a CRLF pair. If the next character is a LF (U+000A)
        // then skip it, and then (unconditionally) return the buffered line.
        if (!m_buffer.isEmpty()) {
            if (m_buffer.currentCharacter() == newlineCharacter)
                m_buffer.advancePastNewline();
            m_maybeSkipLF = false;
        }
        // If there was no (new) data available, then keep m_maybeSkipLF set,
        // and fall through all the way down to the EOS check at the end of the function.
    }

    bool shouldReturnLine = false;
    bool checkForLF = false;
    while (!m_buffer.isEmpty()) {
        UChar character = m_buffer.currentCharacter();
        m_buffer.advance();

        if (character == newlineCharacter || character == carriageReturn) {
            // We found a line ending. Return the accumulated line.
            shouldReturnLine = true;
            checkForLF = (character == carriageReturn);
            break;
        }

        // NULs are transformed into U+FFFD (REPLACEMENT CHAR.) in step 1 of
        // the WebVTT parser algorithm.
        if (character == '\0')
            character = replacementCharacter;

        m_lineBuffer.append(character);
    }

    if (checkForLF) {
        // May be in the middle of a CRLF pair.
        if (!m_buffer.isEmpty()) {
            if (m_buffer.currentCharacter() == newlineCharacter)
                m_buffer.advancePastNewline();
        } else {
            // Check for the newline on the next call (unless we reached EOS, in
            // which case we'll return the contents of the line buffer, and
            // reset state for the next line.)
            m_maybeSkipLF = true;
        }
    }

    if (isAtEndOfStream()) {
        // We've reached the end of the stream proper. Emit a line if the
        // current line buffer is non-empty. (Note that if shouldReturnLine is
        // set already, we want to return a line nonetheless.)
        shouldReturnLine |= !m_lineBuffer.isEmpty();
    }

    if (shouldReturnLine) {
        auto line = m_lineBuffer.toString();
        m_lineBuffer.clear();
        return line;
    }

    ASSERT(m_buffer.isEmpty());
    return std::nullopt;
}

} // namespace WebCore
