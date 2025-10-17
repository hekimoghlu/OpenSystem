/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 31, 2022.
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

#include <optional>

namespace WebCore {

struct TextBoxSelectableRange {
    const unsigned start;
    const unsigned length;
    const unsigned additionalLengthAtEnd { 0 };
    const bool isLineBreak { false };
    // FIXME: Consider holding onto the truncation position instead. See webkit.org/b/164999
    const std::optional<unsigned> truncation { };

    unsigned clamp(unsigned offset) const
    {
        auto clampedOffset = std::clamp(offset, start, start + length) - start;

        // FIXME: For some reason we allow the caret move to (invisible) fully truncated text. The zero test is to keep that behavior.
        if (truncation && *truncation)
            return std::min<unsigned>(clampedOffset, *truncation);

        if (clampedOffset == length)
            clampedOffset += additionalLengthAtEnd;

        return clampedOffset;
    }

    std::pair<unsigned, unsigned> clamp(unsigned startOffset, unsigned endOffset) const
    {
        return { clamp(startOffset), clamp(endOffset) };
    }

    bool intersects(unsigned startOffset, unsigned endOffset) const
    {
        return clamp(startOffset) < clamp(endOffset);
    }
};

}
