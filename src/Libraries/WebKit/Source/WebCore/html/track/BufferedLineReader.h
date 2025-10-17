/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 6, 2023.
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
#pragma once

#include "SegmentedString.h"
#include <wtf/text/StringBuilder.h>

namespace WebCore {

// Line collection helper for the WebVTT Parser.
//
// Converts a stream of data (== a sequence of Strings) into a set of
// lines. CR, LR or CRLF are considered line breaks. Normalizes NULs (U+0000)
// to 'REPLACEMENT CHARACTER' (U+FFFD) and does not return the line breaks as
// part of the result.
class BufferedLineReader {
    WTF_MAKE_NONCOPYABLE(BufferedLineReader);
public:
    BufferedLineReader() = default;
    void reset();

    void append(String&& data)
    {
        ASSERT(!m_endOfStream);
        m_buffer.append(WTFMove(data));
    }

    void appendEndOfStream() { m_endOfStream = true; }
    bool isAtEndOfStream() const { return m_endOfStream && m_buffer.isEmpty(); }

    std::optional<String> nextLine();

private:
    SegmentedString m_buffer;
    StringBuilder m_lineBuffer;
    bool m_endOfStream { false };
    bool m_maybeSkipLF { false };
};

inline void BufferedLineReader::reset()
{
    m_buffer.clear();
    m_lineBuffer.clear();
    m_endOfStream = false;
    m_maybeSkipLF = false;
}

} // namespace WebCore
