/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 30, 2024.
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
#include "AccessibilityLabel.h"

namespace WebCore {
    
using namespace HTMLNames;

AccessibilityLabel::AccessibilityLabel(AXID axID, RenderObject& renderer)
    : AccessibilityRenderObject(axID, renderer)
{
}

AccessibilityLabel::~AccessibilityLabel() = default;

Ref<AccessibilityLabel> AccessibilityLabel::create(AXID axID, RenderObject& renderer)
{
    return adoptRef(*new AccessibilityLabel(axID, renderer));
}

String AccessibilityLabel::stringValue() const
{
    if (containsOnlyStaticText())
        return textUnderElement();
    return AccessibilityNodeObject::stringValue();
}

static bool childrenContainOnlyStaticText(const AccessibilityObject::AccessibilityChildrenVector& children)
{
    if (children.isEmpty())
        return false;
    for (const auto& child : children) {
        if (child->roleValue() == AccessibilityRole::StaticText)
            continue;
        if (child->isGroup()) {
            if (!childrenContainOnlyStaticText(child->unignoredChildren()))
                return false;
        } else
            return false;
    }
    return true;
}

bool AccessibilityLabel::containsOnlyStaticText() const
{
    // m_containsOnlyStaticTextDirty is set (if necessary) by addChildren(), so update our children before checking the flag.
    const_cast<AccessibilityLabel*>(this)->updateChildrenIfNecessary();
    if (m_containsOnlyStaticTextDirty) {
        m_containsOnlyStaticTextDirty = false;
        m_containsOnlyStaticText = childrenContainOnlyStaticText(const_cast<AccessibilityLabel*>(this)->unignoredChildren());
    }
    return m_containsOnlyStaticText;
}

void AccessibilityLabel::addChildren()
{
    AccessibilityRenderObject::addChildren();
    m_containsOnlyStaticTextDirty = true;
}

void AccessibilityLabel::clearChildren()
{
    AccessibilityRenderObject::clearChildren();
    m_containsOnlyStaticText = false;
    m_containsOnlyStaticTextDirty = false;
}

} // namespace WebCore
