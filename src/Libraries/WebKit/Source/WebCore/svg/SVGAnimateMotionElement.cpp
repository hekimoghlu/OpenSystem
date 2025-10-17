/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 18, 2024.
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
#include "SVGAnimateMotionElement.h"

#include "AffineTransform.h"
#include "CommonAtomStrings.h"
#include "ElementChildIteratorInlines.h"
#include "LegacyRenderSVGResource.h"
#include "PathTraversalState.h"
#include "RenderLayerModelObject.h"
#include "SVGElementTypeHelpers.h"
#include "SVGImageElement.h"
#include "SVGMPathElement.h"
#include "SVGNames.h"
#include "SVGParserUtilities.h"
#include "SVGPathData.h"
#include "SVGPathElement.h"
#include "SVGPathUtilities.h"
#include <wtf/MathExtras.h>
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/StringView.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(SVGAnimateMotionElement);
    
using namespace SVGNames;

inline SVGAnimateMotionElement::SVGAnimateMotionElement(const QualifiedName& tagName, Document& document)
    : SVGAnimationElement(tagName, document)
{
    setCalcMode(CalcMode::Paced);
    ASSERT(hasTagName(animateMotionTag));
}

Ref<SVGAnimateMotionElement> SVGAnimateMotionElement::create(const QualifiedName& tagName, Document& document)
{
    return adoptRef(*new SVGAnimateMotionElement(tagName, document));
}

bool SVGAnimateMotionElement::hasValidAttributeType() const
{
    RefPtr targetElement = this->targetElement();
    if (!targetElement)
        return false;

    // We don't have a special attribute name to verify the animation type. Check the element name instead.
    if (!targetElement->isSVGGraphicsElement())
        return false;
    // Spec: SVG 1.1 section 19.2.15
    // FIXME: svgTag is missing. Needs to be checked, if transforming <svg> could cause problems.
    if (targetElement->hasTagName(gTag)
        || targetElement->hasTagName(defsTag)
        || targetElement->hasTagName(useTag)
        || is<SVGImageElement>(*targetElement)
        || targetElement->hasTagName(switchTag)
        || targetElement->hasTagName(pathTag)
        || targetElement->hasTagName(rectTag)
        || targetElement->hasTagName(circleTag)
        || targetElement->hasTagName(ellipseTag)
        || targetElement->hasTagName(lineTag)
        || targetElement->hasTagName(polylineTag)
        || targetElement->hasTagName(polygonTag)
        || targetElement->hasTagName(textTag)
        || targetElement->hasTagName(clipPathTag)
        || targetElement->hasTagName(maskTag)
        || targetElement->hasTagName(SVGNames::aTag)
        || targetElement->hasTagName(foreignObjectTag)
        )
        return true;
    return false;
}

bool SVGAnimateMotionElement::hasValidAttributeName() const
{
    // AnimateMotion does not use attributeName so it is always valid.
    return true;
}

void SVGAnimateMotionElement::attributeChanged(const QualifiedName& name, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason attributeModificationReason)
{
    if (name == SVGNames::pathAttr) {
        m_path = buildPathFromString(newValue);
        updateAnimationPath();
    }

    SVGAnimationElement::attributeChanged(name, oldValue, newValue, attributeModificationReason);
}
    
SVGAnimateMotionElement::RotateMode SVGAnimateMotionElement::rotateMode() const
{
    static MainThreadNeverDestroyed<const AtomString> autoReverse("auto-reverse"_s);
    auto& rotate = getAttribute(SVGNames::rotateAttr);
    if (rotate == autoAtom())
        return RotateAuto;
    if (rotate == autoReverse)
        return RotateAutoReverse;
    return RotateAngle;
}

void SVGAnimateMotionElement::updateAnimationPath()
{
    m_animationPath = Path();
    bool foundMPath = false;

    for (auto& mPath : childrenOfType<SVGMPathElement>(*this)) {
        if (RefPtr pathElement = mPath.pathElement()) {
            m_animationPath = pathFromGraphicsElement(*pathElement);
            foundMPath = true;
            break;
        }
    }

    if (!foundMPath && hasAttributeWithoutSynchronization(SVGNames::pathAttr))
        m_animationPath = m_path;

    updateAnimationMode();
}

void SVGAnimateMotionElement::startAnimation()
{
    if (!hasValidAttributeType())
        return;
    RefPtr targetElement = this->targetElement();
    if (!targetElement)
        return;
    if (auto* transform = targetElement->ensureSupplementalTransform())
        transform->makeIdentity();
}

void SVGAnimateMotionElement::stopAnimation(SVGElement* targetElement)
{
    if (!targetElement)
        return;
    if (auto* transform = targetElement->ensureSupplementalTransform())
        transform->makeIdentity();
    applyResultsToTarget();
}

bool SVGAnimateMotionElement::setFromAndToValues(const String& fromString, const String& toString)
{
    m_toPointAtEndOfDuration = std::nullopt;
    m_fromPoint = valueOrDefault(parsePoint(fromString));
    m_toPoint = valueOrDefault(parsePoint(toString));
    return true;
}
    
bool SVGAnimateMotionElement::setFromAndByValues(const String& fromString, const String& byString)
{
    m_toPointAtEndOfDuration = std::nullopt;
    if (animationMode() == AnimationMode::By && !isAdditive())
        return false;
    m_fromPoint = valueOrDefault(parsePoint(fromString));
    auto byPoint = valueOrDefault(parsePoint(byString));
    m_toPoint = FloatPoint(m_fromPoint.x() + byPoint.x(), m_fromPoint.y() + byPoint.y());
    return true;
}

bool SVGAnimateMotionElement::setToAtEndOfDurationValue(const String& toAtEndOfDurationString)
{
    m_toPointAtEndOfDuration = valueOrDefault(parsePoint(toAtEndOfDurationString));
    return true;
}

void SVGAnimateMotionElement::buildTransformForProgress(AffineTransform* transform, float percentage)
{
    ASSERT(!m_animationPath.isEmpty());

    float positionOnPath = m_animationPath.length() * percentage;
    auto traversalState(m_animationPath.traversalStateAtLength(positionOnPath));
    if (!traversalState.success())
        return;

    FloatPoint position = traversalState.current();
    transform->translate(position);
}

void SVGAnimateMotionElement::calculateAnimatedValue(float percentage, unsigned repeatCount)
{
    RefPtr targetElement = this->targetElement();
    if (!targetElement)
        return;

    auto* transform = targetElement->ensureSupplementalTransform();
    if (!transform)
        return;

    if (!isAdditive())
        transform->makeIdentity();

    if (animationMode() != AnimationMode::Path) {
        FloatPoint toPointAtEndOfDuration = m_toPoint;
        if (isAccumulated() && repeatCount && m_toPointAtEndOfDuration)
            toPointAtEndOfDuration = *m_toPointAtEndOfDuration;

        float animatedX = 0;
        animateAdditiveNumber(percentage, repeatCount, m_fromPoint.x(), m_toPoint.x(), toPointAtEndOfDuration.x(), animatedX);

        float animatedY = 0;
        animateAdditiveNumber(percentage, repeatCount, m_fromPoint.y(), m_toPoint.y(), toPointAtEndOfDuration.y(), animatedY);

        transform->translate(animatedX, animatedY);
        return;
    }

    buildTransformForProgress(transform, percentage);

    // Handle accumulate="sum".
    if (isAccumulated() && repeatCount) {
        for (unsigned i = 0; i < repeatCount; ++i)
            buildTransformForProgress(transform, 1);
    }
    float positionOnPath = m_animationPath.length() * percentage;
    auto traversalState(m_animationPath.traversalStateAtLength(positionOnPath));

    // The 'angle' below is in 'degrees'.
    float angle = traversalState.normalAngle();
    RotateMode rotateMode = this->rotateMode();
    if (rotateMode != RotateAuto && rotateMode != RotateAutoReverse)
        return;
    if (rotateMode == RotateAutoReverse)
        angle += 180;
    transform->rotate(angle);
}

void SVGAnimateMotionElement::applyResultsToTarget()
{
    // We accumulate to the target element transform list so there is not much to do here.
    RefPtr targetElement = this->targetElement();
    if (!targetElement)
        return;

    auto updateTargetElement = [](SVGElement& element) {
        if (element.document().settings().layerBasedSVGEngineEnabled()) {
            if (CheckedPtr layerRenderer = dynamicDowncast<RenderLayerModelObject>(element.renderer()))
                layerRenderer->updateHasSVGTransformFlags();
            // TODO: [LBSE] Avoid relayout upon transform changes (not possible in legacy, but should be in LBSE).
            element.updateSVGRendererForElementChange();
            return;
        }
        if (CheckedPtr renderer = element.renderer())
            renderer->setNeedsTransformUpdate();
        element.updateSVGRendererForElementChange();
    };

    updateTargetElement(*targetElement);

    auto* targetSupplementalTransform = targetElement->ensureSupplementalTransform();
    if (!targetSupplementalTransform)
        return;

    // ...except in case where we have additional instances in <use> trees.
    for (auto& instance : copyToVectorOf<Ref<SVGElement>>(targetElement->instances())) {
        auto* transform = instance->ensureSupplementalTransform();
        if (!transform || *transform == *targetSupplementalTransform)
            continue;
        *transform = *targetSupplementalTransform;
        updateTargetElement(instance);
    }
}

std::optional<float> SVGAnimateMotionElement::calculateDistance(const String& fromString, const String& toString)
{
    auto from = parsePoint(fromString);
    if (!from)
        return { };
    auto to = parsePoint(toString);
    if (!to)
        return { };
    return (*to - *from).diagonalLength();
}

void SVGAnimateMotionElement::updateAnimationMode()
{
    if (!m_animationPath.isEmpty())
        setAnimationMode(AnimationMode::Path);
    else
        SVGAnimationElement::updateAnimationMode();
}

void SVGAnimateMotionElement::childrenChanged(const ChildChange& change)
{
    SVGElement::childrenChanged(change);
    switch (change.type) {
    case ChildChange::Type::ElementRemoved:
    case ChildChange::Type::AllChildrenRemoved:
    case ChildChange::Type::AllChildrenReplaced:
        updateAnimationPath();
        break;
    case ChildChange::Type::ElementInserted:
    case ChildChange::Type::TextInserted:
    case ChildChange::Type::TextRemoved:
    case ChildChange::Type::TextChanged:
    case ChildChange::Type::NonContentsChildInserted:
    case ChildChange::Type::NonContentsChildRemoved:
        break;
    }
}

}
