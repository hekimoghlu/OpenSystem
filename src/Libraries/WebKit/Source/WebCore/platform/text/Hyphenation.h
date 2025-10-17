/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 15, 2025.
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
#ifndef Hyphenation_h
#define Hyphenation_h

#include <unicode/utypes.h>
#include <wtf/Forward.h>

namespace WebCore {

inline static bool enoughWidthForHyphenation(float availableWidth, float fontSize)
{
    // If the maximum width available for the prefix before the hyphen is small, then it is very unlikely
    // that an hyphenation opportunity exists, so do not bother to look for it.
    return availableWidth > fontSize * 5 / 4;

}
bool canHyphenate(const AtomString& localeIdentifier);
size_t lastHyphenLocation(StringView, size_t beforeIndex, const AtomString& localeIdentifier);

} // namespace WebCore

#endif // Hyphenation_h
