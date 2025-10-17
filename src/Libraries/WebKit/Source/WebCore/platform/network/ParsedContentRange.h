/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 14, 2021.
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

#include <wtf/Forward.h>
#include <wtf/Markable.h>

namespace WebCore {

class ParsedContentRange {
public:
    static constexpr int64_t invalidLength = std::numeric_limits<int64_t>::min();
    static constexpr int64_t unknownLength = std::numeric_limits<int64_t>::max();

    WEBCORE_EXPORT explicit ParsedContentRange(const String&);
    WEBCORE_EXPORT ParsedContentRange(int64_t firstBytePosition, int64_t lastBytePosition, int64_t instanceLength);
    ParsedContentRange() = default;

    bool isValid() const { return m_instanceLength != invalidLength; }
    int64_t firstBytePosition() const { return m_firstBytePosition; }
    int64_t lastBytePosition() const { return m_lastBytePosition; }
    int64_t instanceLength() const { return m_instanceLength; }

    static ParsedContentRange invalidValue()
    {
        return ParsedContentRange();
    }

    WEBCORE_EXPORT String headerValue() const;

    struct MarkableTraits {
        static bool isEmptyValue(const ParsedContentRange& range)
        {
            return !range.isValid();
        }

        static ParsedContentRange emptyValue()
        {
            return ParsedContentRange::invalidValue();
        }
    };

private:
    int64_t m_firstBytePosition { 0 };
    int64_t m_lastBytePosition { 0 };
    int64_t m_instanceLength { invalidLength };
};

}
