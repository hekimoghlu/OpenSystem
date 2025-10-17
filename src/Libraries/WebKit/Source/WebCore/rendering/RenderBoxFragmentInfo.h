/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 13, 2025.
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

#include "RenderOverflow.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

class RenderBoxFragmentInfo {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(RenderBoxFragmentInfo);
    WTF_MAKE_NONCOPYABLE(RenderBoxFragmentInfo);
public:
    RenderBoxFragmentInfo(LayoutUnit logicalLeft, LayoutUnit logicalWidth, bool isShifted)
        : m_logicalLeft(logicalLeft)
        , m_logicalWidth(logicalWidth)
        , m_isShifted(isShifted)
    { }

    LayoutUnit logicalLeft() const { return m_logicalLeft; }
    LayoutUnit logicalWidth() const { return m_logicalWidth; }
    
    void shiftLogicalLeft(LayoutUnit delta) { m_logicalLeft += delta; m_isShifted = true; }

    bool isShifted() const { return m_isShifted; }

    void createOverflow(const LayoutRect& layoutOverflow, const LayoutRect& visualOverflow) { m_overflow = adoptRef(new RenderOverflow(layoutOverflow, visualOverflow)); }
    RenderOverflow* overflow() const { return m_overflow.get(); }
    void clearOverflow()
    {
        if (m_overflow)
            m_overflow = nullptr;
    }

private:
    LayoutUnit m_logicalLeft;
    LayoutUnit m_logicalWidth;
    bool m_isShifted;
    RefPtr<RenderOverflow> m_overflow;
};

} // namespace WebCore
