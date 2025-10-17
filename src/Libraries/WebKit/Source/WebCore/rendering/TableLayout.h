/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 22, 2025.
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
#include <wtf/FastMalloc.h>
#include <wtf/Noncopyable.h>

namespace WebCore {

class RenderTable;

enum class TableIntrinsics : uint8_t;

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(TableLayout);
class TableLayout {
    WTF_MAKE_NONCOPYABLE(TableLayout);
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(TableLayout);
public:
    explicit TableLayout(RenderTable* table)
        : m_table(table)
    {
    }

    virtual ~TableLayout() = default;

    virtual void computeIntrinsicLogicalWidths(LayoutUnit& minWidth, LayoutUnit& maxWidth, TableIntrinsics) = 0;
    virtual LayoutUnit scaledWidthFromPercentColumns() const { return 0_lu; }
    virtual void applyPreferredLogicalWidthQuirks(LayoutUnit& minWidth, LayoutUnit& maxWidth) const = 0;
    virtual void layout() = 0;

protected:
    // FIXME: Once we enable SATURATED_LAYOUT_ARITHMETHIC, this should just be LayoutUnit::nearlyMax().
    // Until then though, using nearlyMax causes overflow in some tests, so we just pick a large number.
    static constexpr int tableMaxWidth = 1000000;

    RenderTable* m_table;
};

} // namespace WebCore
