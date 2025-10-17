/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 26, 2022.
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

#include "SVGSMILElement.h"
#include "SVGTests.h"
#include "UnitBezier.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class ConditionEventListener;
class TimeContainer;

// If we have 'currentColor' or 'inherit' as animation value, we need to grab
// the value during the animation since the value can be animated itself.
enum AnimatedPropertyValueType { RegularPropertyValue, CurrentColorValue, InheritValue };

class SVGAnimationElement : public SVGSMILElement, public SVGTests {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SVGAnimationElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SVGAnimationElement);
public:
    ExceptionOr<float> getStartTime() const;
    float getCurrentTime() const;
    ExceptionOr<float> getSimpleDuration() const;

    void beginElement() { beginElementAt(0); }
    void beginElementAt(float offset);
    void endElement() { endElementAt(0); }
    void endElementAt(float offset);

    static bool isTargetAttributeCSSProperty(SVGElement*, const QualifiedName&);

    bool isAdditive() const override;
    bool isAccumulated() const;
    AnimationMode animationMode() const { return m_animationMode; }
    CalcMode calcMode() const { return m_calcMode; }

    AnimatedPropertyValueType fromPropertyValueType() const { return m_fromPropertyValueType; }
    AnimatedPropertyValueType toPropertyValueType() const { return m_toPropertyValueType; }

    void animateAdditiveNumber(float percentage, unsigned repeatCount, float fromNumber, float toNumber, float toAtEndOfDurationNumber, float& animatedNumber)
    {
        float number;
        if (calcMode() == CalcMode::Discrete)
            number = percentage < 0.5 ? fromNumber : toNumber;
        else
            number = (toNumber - fromNumber) * percentage + fromNumber;

        if (isAccumulated() && repeatCount)
            number += toAtEndOfDurationNumber * repeatCount;

        if (isAdditive() && animationMode() != AnimationMode::To)
            animatedNumber += number;
        else
            animatedNumber = number;
    }

    enum class AttributeType : uint8_t { CSS, XML, Auto };
    AttributeType attributeType() const { return m_attributeType; }

    using PropertyRegistry = SVGPropertyOwnerRegistry<SVGAnimationElement, SVGElement, SVGTests>;

protected:
    SVGAnimationElement(const QualifiedName&, Document&);

    virtual void resetAnimation();

    static bool isSupportedAttribute(const QualifiedName&);
    bool attributeContainsJavaScriptURL(const Attribute&) const final;
    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) override;
    void svgAttributeChanged(const QualifiedName&) override;

    String toValue() const;
    String byValue() const;
    String fromValue() const;

    // from SVGSMILElement
    void startedActiveInterval() override;
    void updateAnimation(float percent, unsigned repeat) override;

    AnimatedPropertyValueType m_fromPropertyValueType { RegularPropertyValue };
    AnimatedPropertyValueType m_toPropertyValueType { RegularPropertyValue };

    void setAttributeName(const QualifiedName&) override { }

    virtual void updateAnimationMode();
    void setAnimationMode(AnimationMode animationMode) { m_animationMode = animationMode; }
    void setCalcMode(CalcMode calcMode) { m_calcMode = calcMode; }

private:
    void animationAttributeChanged() override;
    void setAttributeType(const AtomString&);

    virtual bool setFromAndToValues(const String& fromString, const String& toString) = 0;
    virtual bool setFromAndByValues(const String& fromString, const String& byString) = 0;
    virtual bool setToAtEndOfDurationValue(const String& toAtEndOfDurationString) = 0;
    virtual void calculateAnimatedValue(float percent, unsigned repeatCount) = 0;
    virtual std::optional<float> calculateDistance(const String& /*fromString*/, const String& /*toString*/) = 0;

    const Vector<float>& keyTimes() const;
    void currentValuesForValuesAnimation(float percent, float& effectivePercent, String& from, String& to);
    void calculateKeyTimesForCalcModePaced();
    float calculatePercentFromKeyPoints(float percent) const;
    void currentValuesFromKeyPoints(float percent, float& effectivePercent, String& from, String& to) const;
    float calculatePercentForSpline(float percent, unsigned splineIndex) const;
    float calculatePercentForFromTo(float percent) const;
    unsigned calculateKeyTimesIndex(float percent) const;

    void setCalcMode(const AtomString&);

    bool m_animationValid { false };

    AttributeType m_attributeType { AttributeType::Auto };
    Vector<String> m_values;
    Vector<float> m_keyTimesFromAttribute;
    Vector<float> m_keyTimesForPaced;
    Vector<float> m_keyPoints;
    Vector<UnitBezier> m_keySplines;
    String m_lastValuesAnimationFrom;
    String m_lastValuesAnimationTo;
    CalcMode m_calcMode { CalcMode::Linear };
    AnimationMode m_animationMode { AnimationMode::None };
};

} // namespace WebCore
