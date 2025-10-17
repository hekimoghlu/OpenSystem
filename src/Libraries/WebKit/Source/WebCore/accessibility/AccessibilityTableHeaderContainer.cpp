/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 28, 2022.
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
#include "AccessibilityTableHeaderContainer.h"

#include "AXObjectCache.h"
#include "AccessibilityTable.h"

namespace WebCore {

AccessibilityTableHeaderContainer::AccessibilityTableHeaderContainer(AXID axID)
    : AccessibilityMockObject(axID)
{
}

AccessibilityTableHeaderContainer::~AccessibilityTableHeaderContainer() = default;

Ref<AccessibilityTableHeaderContainer> AccessibilityTableHeaderContainer::create(AXID axID)
{
    return adoptRef(*new AccessibilityTableHeaderContainer(axID));
}
    
LayoutRect AccessibilityTableHeaderContainer::elementRect() const
{
    // this will be filled in when addChildren is called
    return m_headerRect;
}

bool AccessibilityTableHeaderContainer::computeIsIgnored() const
{
#if PLATFORM(IOS_FAMILY) || USE(ATSPI)
    return true;
#endif
    return !m_parent || m_parent->isIgnored();
}

void AccessibilityTableHeaderContainer::addChildren()
{
    ASSERT(!m_childrenInitialized); 
    
    m_childrenInitialized = true;
    RefPtr parentTable = dynamicDowncast<AccessibilityTable>(m_parent.get());
    if (!parentTable || !parentTable->isExposable())
        return;

    for (auto& columnHeader : parentTable->columnHeaders())
        addChild(downcast<AccessibilityObject>(columnHeader.get()));

    for (const auto& child : m_children)
        m_headerRect.unite(child->elementRect());
}

} // namespace WebCore
