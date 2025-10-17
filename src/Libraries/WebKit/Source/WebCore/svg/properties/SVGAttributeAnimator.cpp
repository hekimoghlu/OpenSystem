/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 5, 2023.
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
#include "SVGAttributeAnimator.h"

#include "CSSComputedStyleDeclaration.h"
#include "CSSPropertyParser.h"
#include "MutableStyleProperties.h"
#include "SVGElementInlines.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(SVGAttributeAnimator);

bool SVGAttributeAnimator::isAnimatedStylePropertyAnimator(const SVGElement& targetElement) const
{
    return targetElement.isAnimatedStyleAttribute(m_attributeName);
}

void SVGAttributeAnimator::invalidateStyle(SVGElement& targetElement)
{
    SVGElement::InstanceInvalidationGuard guard(targetElement);
    targetElement.setPresentationalHintStyleIsDirty();
}

void SVGAttributeAnimator::applyAnimatedStylePropertyChange(SVGElement& element, CSSPropertyID id, const String& value)
{
    ASSERT(!element.deletionHasBegun());
    ASSERT(id != CSSPropertyInvalid);
    
    if (!element.ensureAnimatedSMILStyleProperties().setProperty(id, value))
        return;
    element.invalidateStyle();
}

void SVGAttributeAnimator::applyAnimatedStylePropertyChange(SVGElement& targetElement, const String& value)
{
    ASSERT(m_attributeName != anyQName());
    
    // FIXME: Do we really need to check both isConnected and !parentNode?
    if (!targetElement.isConnected() || !targetElement.parentNode())
        return;
    
    CSSPropertyID id = cssPropertyID(m_attributeName.localName());
    
    SVGElement::InstanceUpdateBlocker blocker(targetElement);
    applyAnimatedStylePropertyChange(targetElement, id, value);
    
    // If the target element has instances, update them as well, w/o requiring the <use> tree to be rebuilt.
    for (auto& instance : copyToVectorOf<Ref<SVGElement>>(targetElement.instances()))
        applyAnimatedStylePropertyChange(instance, id, value);
}
    
void SVGAttributeAnimator::removeAnimatedStyleProperty(SVGElement& element, CSSPropertyID id)
{
    ASSERT(!element.deletionHasBegun());
    ASSERT(id != CSSPropertyInvalid);

    element.ensureAnimatedSMILStyleProperties().removeProperty(id);
    element.invalidateStyle();
}

void SVGAttributeAnimator::removeAnimatedStyleProperty(SVGElement& targetElement)
{
    ASSERT(m_attributeName != anyQName());

    // FIXME: Do we really need to check both isConnected and !parentNode?
    if (!targetElement.isConnected() || !targetElement.parentNode())
        return;

    CSSPropertyID id = cssPropertyID(m_attributeName.localName());

    SVGElement::InstanceUpdateBlocker blocker(targetElement);
    removeAnimatedStyleProperty(targetElement, id);

    // If the target element has instances, update them as well, w/o requiring the <use> tree to be rebuilt.
    for (auto& instance : copyToVectorOf<Ref<SVGElement>>(targetElement.instances()))
        removeAnimatedStyleProperty(instance, id);
}
    
void SVGAttributeAnimator::applyAnimatedPropertyChange(SVGElement& element, const QualifiedName& attributeName)
{
    ASSERT(!element.deletionHasBegun());
    element.svgAttributeChanged(attributeName);
}

void SVGAttributeAnimator::applyAnimatedPropertyChange(SVGElement& targetElement)
{
    ASSERT(m_attributeName != anyQName());

    // FIXME: Do we really need to check both isConnected and !parentNode?
    if (!targetElement.isConnected() || !targetElement.parentNode())
        return;

    SVGElement::InstanceUpdateBlocker blocker(targetElement);
    applyAnimatedPropertyChange(targetElement, m_attributeName);

    // If the target element has instances, update them as well, w/o requiring the <use> tree to be rebuilt.
    for (auto& instance : copyToVectorOf<Ref<SVGElement>>(targetElement.instances()))
        applyAnimatedPropertyChange(instance, m_attributeName);
}

} // namespace WebCore
