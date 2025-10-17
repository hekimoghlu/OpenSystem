/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 2, 2022.
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

#include "CSSPropertyNames.h"
#include "QualifiedName.h"
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class SVGElement;

enum class AnimationMode : uint8_t {
    None,
    FromTo,
    FromBy,
    To,
    By,
    Values,
    Path
};

enum class CalcMode : uint8_t {
    Discrete,
    Linear,
    Paced,
    Spline
};

class SVGAttributeAnimator : public RefCountedAndCanMakeWeakPtr<SVGAttributeAnimator> {
    WTF_MAKE_TZONE_ALLOCATED(SVGAttributeAnimator);
public:
    SVGAttributeAnimator(const QualifiedName& attributeName)
        : m_attributeName(attributeName)
    {
    }

    virtual ~SVGAttributeAnimator() = default;

    virtual bool isDiscrete() const { return false; }

    virtual void setFromAndToValues(SVGElement&, const String&, const String&) { }
    virtual void setFromAndByValues(SVGElement&, const String&, const String&) { }
    virtual void setToAtEndOfDurationValue(const String&) { }

    virtual void start(SVGElement&) = 0;
    virtual void animate(SVGElement&, float progress, unsigned repeatCount) = 0;
    virtual void apply(SVGElement&) = 0;
    virtual void stop(SVGElement& targetElement) = 0;

    virtual std::optional<float> calculateDistance(SVGElement&, const String&, const String&) const { return { }; }

protected:
    bool isAnimatedStylePropertyAnimator(const SVGElement&) const;

    static void invalidateStyle(SVGElement&);
    static void applyAnimatedStylePropertyChange(SVGElement&, CSSPropertyID, const String& value);
    static void removeAnimatedStyleProperty(SVGElement&, CSSPropertyID);
    static void applyAnimatedPropertyChange(SVGElement&, const QualifiedName&);

    void applyAnimatedStylePropertyChange(SVGElement&, const String& value);
    void removeAnimatedStyleProperty(SVGElement&);
    void applyAnimatedPropertyChange(SVGElement&);

    const QualifiedName& m_attributeName;
};

} // namespace WebCore
