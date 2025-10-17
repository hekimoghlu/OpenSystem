/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 27, 2023.
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

#include "RenderStyleInlines.h"
#include "RenderText.h"
#include "WordTrailingSpace.h"

namespace WebCore {

inline LayoutUnit RenderText::marginLeft() const { return minimumValueForLength(style().marginLeft(), 0); }
inline LayoutUnit RenderText::marginRight() const { return minimumValueForLength(style().marginRight(), 0); }

template <typename MeasureTextCallback>
float RenderText::measureTextConsideringPossibleTrailingSpace(bool currentCharacterIsSpace, unsigned startIndex, unsigned wordLength, WordTrailingSpace& wordTrailingSpace, SingleThreadWeakHashSet<const Font>& fallbackFonts, MeasureTextCallback&& callback)
{
    std::optional<float> wordTrailingSpaceWidth;
    if (currentCharacterIsSpace)
        wordTrailingSpaceWidth = wordTrailingSpace.width(fallbackFonts);
    if (wordTrailingSpaceWidth)
        return callback(startIndex, wordLength + 1) - wordTrailingSpaceWidth.value();
    return callback(startIndex, wordLength);
}

} // namespace WebCore
