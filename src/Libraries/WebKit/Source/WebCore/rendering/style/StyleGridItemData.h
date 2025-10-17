/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 31, 2022.
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

#include "GridPosition.h"
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>

namespace WebCore {

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(StyleGridItemData);
class StyleGridItemData : public RefCounted<StyleGridItemData> {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(StyleGridItemData);
public:
    static Ref<StyleGridItemData> create() { return adoptRef(*new StyleGridItemData); }
    Ref<StyleGridItemData> copy() const;

    bool operator==(const StyleGridItemData& o) const
    {
        return gridColumnStart == o.gridColumnStart && gridColumnEnd == o.gridColumnEnd
            && gridRowStart == o.gridRowStart && gridRowEnd == o.gridRowEnd;
    }

#if !LOG_DISABLED
    void dumpDifferences(TextStream&, const StyleGridItemData&) const;
#endif

    GridPosition gridColumnStart;
    GridPosition gridColumnEnd;
    GridPosition gridRowStart;
    GridPosition gridRowEnd;

private:
    StyleGridItemData();
    StyleGridItemData(const StyleGridItemData&);
};

} // namespace WebCore
