/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 24, 2023.
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
#include "RenderTableSection.h"

namespace WebCore {

inline const BorderValue& RenderTableSection::borderAdjoiningTableEnd() const { return style().borderEnd(table()->writingMode()); }
inline const BorderValue& RenderTableSection::borderAdjoiningTableStart() const { return style().borderStart(table()->writingMode()); }

inline LayoutUnit RenderTableSection::outerBorderBottom(const WritingMode writingMode) const
{
    if (writingMode.isHorizontal())
        return writingMode.isBlockTopToBottom() ? outerBorderAfter() : outerBorderBefore();
    return writingMode.isInlineTopToBottom() ? outerBorderEnd() : outerBorderStart();
}

inline LayoutUnit RenderTableSection::outerBorderLeft(const WritingMode writingMode) const
{
    if (writingMode.isHorizontal())
        return writingMode.isInlineLeftToRight() ? outerBorderStart() : outerBorderEnd();
    return writingMode.isBlockLeftToRight() ? outerBorderBefore() : outerBorderAfter();
}

inline LayoutUnit RenderTableSection::outerBorderRight(const WritingMode writingMode) const
{
    if (writingMode.isHorizontal())
        return writingMode.isInlineLeftToRight() ? outerBorderEnd() : outerBorderStart();
    return writingMode.isBlockLeftToRight() ? outerBorderAfter() : outerBorderBefore();
}

inline LayoutUnit RenderTableSection::outerBorderTop(const WritingMode writingMode) const
{
    if (writingMode.isHorizontal())
        return writingMode.isBlockTopToBottom() ? outerBorderBefore() : outerBorderAfter();
    return writingMode.isInlineTopToBottom() ? outerBorderStart() : outerBorderEnd();
}

} // namespace WebCore
