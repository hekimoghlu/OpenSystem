/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 27, 2024.
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

#if USE(CF)
#include <CoreFoundation/CoreFoundation.h>
#endif

#if USE(FOUNDATION)
typedef struct _NSRange NSRange;
#endif

namespace WTF {
class TextStream;
}

namespace WebCore {

struct CharacterRange {
    uint64_t location { 0 };
    uint64_t length { 0 };

    CharacterRange() = default;
    constexpr CharacterRange(uint64_t location, uint64_t length);

    bool operator==(const CharacterRange&) const = default;

#if USE(CF)
    constexpr CharacterRange(CFRange);
    constexpr operator CFRange() const;
#endif

#if USE(FOUNDATION)
    constexpr CharacterRange(NSRange);
    constexpr operator NSRange() const;
#endif
};

constexpr CharacterRange::CharacterRange(uint64_t location, uint64_t length)
    : location(location)
    , length(length)
{
}

WEBCORE_EXPORT WTF::TextStream& operator<<(WTF::TextStream&, const CharacterRange&);

#if USE(CF)

constexpr CharacterRange::CharacterRange(CFRange range)
    : CharacterRange(range.location, range.length)
{
    ASSERT(range.location != kCFNotFound);
}

constexpr CharacterRange::operator CFRange() const
{
    CFIndex locationCF = location;
    CFIndex lengthCF = length;
    return { locationCF, lengthCF };
}

#endif

#if USE(FOUNDATION) && defined(__OBJC__)

constexpr CharacterRange::CharacterRange(NSRange range)
    : CharacterRange { range.location, range.length }
{
    ASSERT(range.location != NSNotFound);
}

constexpr CharacterRange::operator NSRange() const
{
    NSUInteger locationNS = location;
    NSUInteger lengthNS = length;
    return { locationNS, lengthNS };
}

#endif

} // namespace WebCore
