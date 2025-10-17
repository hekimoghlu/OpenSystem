/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 17, 2025.
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
#include "config.h"
#include "AccessibilityScrollbar.h"

#include "AXObjectCache.h"
#include "LocalFrameView.h"
#include "ScrollView.h"
#include "Scrollbar.h"

namespace WebCore {

AccessibilityScrollbar::AccessibilityScrollbar(AXID axID, Scrollbar& scrollbar)
    : AccessibilityMockObject(axID)
    , m_scrollbar(scrollbar)
{
}

Ref<AccessibilityScrollbar> AccessibilityScrollbar::create(AXID axID, Scrollbar& scrollbar)
{
    return adoptRef(*new AccessibilityScrollbar(axID, scrollbar));
}
    
LayoutRect AccessibilityScrollbar::elementRect() const
{
    return m_scrollbar->frameRect();
}
    
Document* AccessibilityScrollbar::document() const
{
    RefPtr parent = parentObject();
    return parent ? parent->document() : nullptr;
}

AccessibilityOrientation AccessibilityScrollbar::orientation() const
{
    // ARIA 1.1 Elements with the role scrollbar have an implicit aria-orientation value of vertical.
    if (m_scrollbar->orientation() == ScrollbarOrientation::Horizontal)
        return AccessibilityOrientation::Horizontal;
    if (m_scrollbar->orientation() == ScrollbarOrientation::Vertical)
        return AccessibilityOrientation::Vertical;

    return AccessibilityOrientation::Vertical;
}

bool AccessibilityScrollbar::isEnabled() const
{
    return m_scrollbar->enabled();
}
    
float AccessibilityScrollbar::valueForRange() const
{
    return m_scrollbar->currentPos() / m_scrollbar->maximum();
}

bool AccessibilityScrollbar::setValue(float value)
{
    float newValue = value * m_scrollbar->maximum();
    m_scrollbar->scrollableArea().scrollToOffsetWithoutAnimation(m_scrollbar->orientation(), newValue);
    return true;
}
    
} // namespace WebCore
