/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 20, 2023.
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

#include "RenderBox.h"
#include <wtf/CheckedPtr.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

enum class ItemLayoutRequirement : uint8_t {
    NeedsColumnAxisStretchAlignment = 1 << 0,
    MinContentContributionForSecondColumnPass = 1 << 1,
};
using ItemsLayoutRequirements = SingleThreadWeakHashMap<RenderBox, OptionSet<ItemLayoutRequirement>>;

class GridLayoutState {
    WTF_MAKE_TZONE_ALLOCATED(GridLayoutState);
public:
    bool containsLayoutRequirementForGridItem(const RenderBox& gridItem, ItemLayoutRequirement) const;
    void setLayoutRequirementForGridItem(const RenderBox& gridItem, ItemLayoutRequirement);

    bool needsSecondTrackSizingPass() const { return m_needsSecondTrackSizingPass; }
    void setNeedsSecondTrackSizingPass() { m_needsSecondTrackSizingPass = true; }

private:
    ItemsLayoutRequirements m_itemsLayoutRequirements;
    bool m_needsSecondTrackSizingPass { false };
};

} // namespace WebCore
