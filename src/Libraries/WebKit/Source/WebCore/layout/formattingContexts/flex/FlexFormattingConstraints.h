/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 5, 2022.
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

namespace WebCore {
namespace Layout {

struct ConstraintsForFlexContent {
    struct AxisGeometry {
        std::optional<LayoutUnit> minimumSize;
        std::optional<LayoutUnit> maximumSize;
        std::optional<LayoutUnit> availableSize;
        LayoutUnit startPosition;
    };
    ConstraintsForFlexContent(const AxisGeometry& mainAxis, const AxisGeometry& crossAxis, bool isSizedUnderMinMax);
    const AxisGeometry& mainAxis() const { return m_mainAxisGeometry; }
    const AxisGeometry& crossAxis() const { return m_crossAxisGeometry; }
    bool isSizedUnderMinMax() const { return m_isSizedUnderMinMax; }

private:
    AxisGeometry m_mainAxisGeometry;
    AxisGeometry m_crossAxisGeometry;
    bool m_isSizedUnderMinMax { false };
};

inline ConstraintsForFlexContent::ConstraintsForFlexContent(const AxisGeometry& mainAxis, const AxisGeometry& crossAxis, bool isSizedUnderMinMax)
    : m_mainAxisGeometry(mainAxis)
    , m_crossAxisGeometry(crossAxis)
    , m_isSizedUnderMinMax(isSizedUnderMinMax)
{
}

}
}

