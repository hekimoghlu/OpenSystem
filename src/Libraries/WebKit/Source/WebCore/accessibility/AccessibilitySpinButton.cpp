/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 27, 2024.
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
#include "AccessibilitySpinButton.h"

#include "AXObjectCache.h"
#include "RenderElement.h"

namespace WebCore {

Ref<AccessibilitySpinButton> AccessibilitySpinButton::create(AXID axID, AXObjectCache& cache)
{
    return adoptRef(*new AccessibilitySpinButton(axID, cache));
}
    
AccessibilitySpinButton::AccessibilitySpinButton(AXID axID, AXObjectCache& cache)
    : AccessibilityMockObject(axID)
    , m_spinButtonElement(nullptr)
    , m_incrementor(downcast<AccessibilitySpinButtonPart>(*cache.create(AccessibilityRole::SpinButtonPart)))
    , m_decrementor(downcast<AccessibilitySpinButtonPart>(*cache.create(AccessibilityRole::SpinButtonPart)))
{
    m_incrementor->setIsIncrementor(true);
    m_incrementor->setParent(this);

    m_decrementor->setIsIncrementor(false);
    m_decrementor->setParent(this);

    addChild(m_incrementor.get());
    addChild(m_decrementor.get());
    m_childrenInitialized = true;
}

AccessibilitySpinButton::~AccessibilitySpinButton() = default;
    
AccessibilitySpinButtonPart* AccessibilitySpinButton::incrementButton()
{
    ASSERT(m_childrenInitialized);
    RELEASE_ASSERT(m_children.size() == 2);
    return &downcast<AccessibilitySpinButtonPart>(m_children[0].get());
}
   
AccessibilitySpinButtonPart* AccessibilitySpinButton::decrementButton()
{
    ASSERT(m_childrenInitialized);
    RELEASE_ASSERT(m_children.size() == 2);
    return &downcast<AccessibilitySpinButtonPart>(m_children[1].get());
}
    
LayoutRect AccessibilitySpinButton::elementRect() const
{
    ASSERT(m_spinButtonElement);
    
    CheckedPtr renderer = m_spinButtonElement ? m_spinButtonElement->renderer() : nullptr;
    if (!renderer)
        return { };

    Vector<FloatQuad> quads;
    renderer->absoluteFocusRingQuads(quads);
    return boundingBoxForQuads(renderer.get(), quads);
}

void AccessibilitySpinButton::addChildren()
{
    // This class sets its children once in the constructor, and should never
    // have dirty or uninitialized children afterwards.
    ASSERT(m_childrenInitialized);
    ASSERT(!m_subtreeDirty);
    ASSERT(!m_childrenDirty);
}
    
void AccessibilitySpinButton::step(int amount)
{
    ASSERT(m_spinButtonElement);
    if (m_spinButtonElement)
        m_spinButtonElement->step(amount);
}

} // namespace WebCore
