/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 27, 2023.
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

#include "Length.h"

namespace WTF {
class TextStream;
}

namespace WebCore {

// This class wraps the length of a gap used by properties column-gap and row-gap.
// The valid values are "normal" or a non-negative <length-percentage>.
class GapLength {
public:
    GapLength()
        : m_isNormal(true)
    {
    }

    GapLength(const Length& length)
        : m_isNormal(false)
        , m_length(length)
    {
    }

    bool isNormal() const { return m_isNormal; }
    const Length& length() const { ASSERT(!m_isNormal); return m_length; }

    friend bool operator==(const GapLength&, const GapLength&) = default;

private:
    bool m_isNormal;
    Length m_length;
};

WTF::TextStream& operator<<(WTF::TextStream&, const GapLength&);

} // namespace WebCore
