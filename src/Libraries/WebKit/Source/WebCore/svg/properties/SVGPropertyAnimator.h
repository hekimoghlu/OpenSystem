/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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

#include "CSSPropertyParser.h"
#include "ComputedStyleExtractor.h"
#include "SVGAttributeAnimator.h"
#include "SVGElement.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {
    
template<typename AnimationFunction>
class SVGPropertyAnimator : public SVGAttributeAnimator {
    WTF_MAKE_TZONE_ALLOCATED_TEMPLATE(SVGPropertyAnimator);
public:
    bool isDiscrete() const override { return m_function.isDiscrete(); }

    void setFromAndToValues(SVGElement& targetElement, const String& from, const String& to) override
    {
        m_function.setFromAndToValues(targetElement, adjustForInheritance(targetElement, from), adjustForInheritance(targetElement, to));
    }

    void setFromAndByValues(SVGElement& targetElement, const String& from, const String& by) override
    {
        m_function.setFromAndByValues(targetElement, from, by);
    }

    void setToAtEndOfDurationValue(const String& toAtEndOfDuration) override
    {
        m_function.setToAtEndOfDurationValue(toAtEndOfDuration);
    }

protected:
    template<typename... Arguments>
    SVGPropertyAnimator(const QualifiedName& attributeName, Arguments&&... arguments)
        : SVGAttributeAnimator(attributeName)
        , m_function(std::forward<Arguments>(arguments)...)
    {
    }

    void stop(SVGElement& targetElement) override
    {
        removeAnimatedStyleProperty(targetElement);
    }

    std::optional<float> calculateDistance(SVGElement& targetElement, const String& from, const String& to) const override
    {
        return m_function.calculateDistance(targetElement, from, to);
    }

    String adjustForInheritance(SVGElement& targetElement, const String& value) const
    {
        static MainThreadNeverDestroyed<const AtomString> inherit("inherit"_s);
        return value == inherit ? computeInheritedCSSPropertyValue(targetElement) : value;
    }

    String computeCSSPropertyValue(SVGElement& targetElement, CSSPropertyID id) const
    {
        Ref protector = targetElement;

        // Don't include any properties resulting from CSS Transitions/Animations or SMIL animations, as we want to retrieve the "base value".
        targetElement.setUseOverrideComputedStyle(true);
        RefPtr<CSSValue> value = ComputedStyleExtractor(&targetElement).propertyValue(id);
        targetElement.setUseOverrideComputedStyle(false);
        return value ? value->cssText() : String();
    }

    String computeInheritedCSSPropertyValue(SVGElement& targetElement) const
    {
        RefPtr svgParent = dynamicDowncast<SVGElement>(targetElement.parentElement());
        if (!svgParent)
            return emptyString();
        return computeCSSPropertyValue(*svgParent, cssPropertyID(m_attributeName.localName()));
    }

    AnimationFunction m_function;
};

WTF_MAKE_TZONE_ALLOCATED_TEMPLATE_IMPL(template<typename AnimationFunction>, SVGPropertyAnimator<AnimationFunction>);

} // namespace WebCore
