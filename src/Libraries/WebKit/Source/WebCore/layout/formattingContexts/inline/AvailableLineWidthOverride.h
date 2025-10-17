/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 10, 2023.
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

#include "LayoutUnit.h"
#include <optional>
#include <wtf/Vector.h>

namespace WebCore {
namespace Layout {

// This class overrides the total line width available for candidate content on a line.
// Note that this only impacts how much content can fit in a line, and it does not change
// the line box dimensions themselves (i.e. this won't change where text is aligned, etc).
class AvailableLineWidthOverride {
public:
    AvailableLineWidthOverride() = default;
    AvailableLineWidthOverride(LayoutUnit globalLineWidthOverride) { m_globalLineWidthOverride = globalLineWidthOverride; }
    AvailableLineWidthOverride(Vector<LayoutUnit> individualLineWidthOverrides) { m_individualLineWidthOverrides = individualLineWidthOverrides; }
    std::optional<LayoutUnit> availableLineWidthOverrideForLine(size_t lineIndex) const
    {
        if (m_globalLineWidthOverride)
            return m_globalLineWidthOverride;
        if (m_individualLineWidthOverrides && lineIndex < m_individualLineWidthOverrides->size())
            return m_individualLineWidthOverrides.value()[lineIndex];
        return { };
    }

private:
    // Logical width constraint applied to all lines
    // Takes precedence over individual width overrides
    std::optional<LayoutUnit> m_globalLineWidthOverride;

    // Logical width constraints applied separately for each line
    std::optional<Vector<LayoutUnit>> m_individualLineWidthOverrides;
};

}
}
