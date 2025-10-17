/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 11, 2025.
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
#include "SVGAnimateElementBase.h"

#include "QualifiedName.h"
#include "SVGAttributeAnimator.h"
#include "SVGElement.h"
#include "SVGNames.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SVGAnimateElementBase);

SVGAnimateElementBase::SVGAnimateElementBase(const QualifiedName& tagName, Document& document)
    : SVGAnimationElement(tagName, document)
{
    ASSERT(hasTagName(SVGNames::animateTag)
        || hasTagName(SVGNames::setTag)
        || hasTagName(SVGNames::animateTransformTag));
}

SVGAttributeAnimator* SVGAnimateElementBase::animator() const
{
    ASSERT(targetElement());
    ASSERT(!hasInvalidCSSAttributeType());

    if (!m_animator)
        m_animator = protectedTargetElement()->createAnimator(attributeName(), animationMode(), calcMode(), isAccumulated(), isAdditive());

    return m_animator.get();
}

bool SVGAnimateElementBase::hasValidAttributeType() const
{
    if (!targetElement() || hasInvalidCSSAttributeType())
        return false;

    return protectedTargetElement()->isAnimatedAttribute(attributeName());
}

bool SVGAnimateElementBase::hasInvalidCSSAttributeType() const
{
    if (!targetElement())
        return false;

    if (!m_hasInvalidCSSAttributeType)
        m_hasInvalidCSSAttributeType = hasValidAttributeName() && attributeType() == AttributeType::CSS && !isTargetAttributeCSSProperty(protectedTargetElement().get(), attributeName());

    return m_hasInvalidCSSAttributeType.value();
}

bool SVGAnimateElementBase::isDiscreteAnimator() const
{
    if (!hasValidAttributeType())
        return false;

    RefPtr animator = this->animator();
    return animator && animator->isDiscrete();
}

void SVGAnimateElementBase::setTargetElement(SVGElement* targetElement)
{
    SVGAnimationElement::setTargetElement(targetElement);
    resetAnimation();
}

void SVGAnimateElementBase::setAttributeName(const QualifiedName& attributeName)
{
    SVGSMILElement::setAttributeName(attributeName);
    resetAnimation();
}

void SVGAnimateElementBase::resetAnimation()
{
    SVGAnimationElement::resetAnimation();
    m_animator = nullptr;
    m_hasInvalidCSSAttributeType = { };
}

bool SVGAnimateElementBase::setFromAndToValues(const String& fromString, const String& toString)
{
    if (!targetElement())
        return false;

    if (RefPtr animator = this->animator()) {
        animator->setFromAndToValues(*protectedTargetElement(), animateRangeString(fromString), animateRangeString(toString));
        return true;
    }
    return false;
}

bool SVGAnimateElementBase::setFromAndByValues(const String& fromString, const String& byString)
{
    if (!targetElement())
        return false;

    if (animationMode() == AnimationMode::By && (!isAdditive() || isDiscreteAnimator()))
        return false;

    if (animationMode() == AnimationMode::FromBy && isDiscreteAnimator())
        return false;

    if (RefPtr animator = this->animator()) {
        animator->setFromAndByValues(*protectedTargetElement(), animateRangeString(fromString), animateRangeString(byString));
        return true;
    }
    return false;
}

bool SVGAnimateElementBase::setToAtEndOfDurationValue(const String& toAtEndOfDurationString)
{
    if (!targetElement() || toAtEndOfDurationString.isEmpty())
        return false;

    if (isDiscreteAnimator())
        return true;

    if (RefPtr animator = this->animator()) {
        animator->setToAtEndOfDurationValue(animateRangeString(toAtEndOfDurationString));
        return true;
    }
    return false;
}

void SVGAnimateElementBase::startAnimation()
{
    if (!targetElement())
        return;

    if (RefPtr protectedAnimator = this->animator())
        protectedAnimator->start(*protectedTargetElement());
}

void SVGAnimateElementBase::calculateAnimatedValue(float progress, unsigned repeatCount)
{
    if (!targetElement())
        return;

    ASSERT(progress >= 0 && progress <= 1);
    if (hasTagName(SVGNames::setTag))
        progress = 1;

    if (calcMode() == CalcMode::Discrete)
        progress = progress < 0.5 ? 0 : 1;

    if (RefPtr animator = this->animator())
        animator->animate(*protectedTargetElement(), progress, repeatCount);
}

void SVGAnimateElementBase::applyResultsToTarget()
{
    if (!targetElement())
        return;

    if (RefPtr animator = this->animator())
        animator->apply(*protectedTargetElement());
}

void SVGAnimateElementBase::stopAnimation(SVGElement* targetElement)
{
    if (!targetElement)
        return;

    if (RefPtr animator = this->animatorIfExists())
        animator->stop(*targetElement);
}

std::optional<float> SVGAnimateElementBase::calculateDistance(const String& fromString, const String& toString)
{
    // FIXME: A return value of float is not enough to support paced animations on lists.
    if (!targetElement())
        return { };

    if (RefPtr animator = this->animator())
        return animator->calculateDistance(*protectedTargetElement(), fromString, toString);

    return { };
}

} // namespace WebCore
