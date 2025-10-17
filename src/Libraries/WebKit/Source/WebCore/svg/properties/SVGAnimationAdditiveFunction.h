/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 6, 2024.
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

#include "SVGAnimationFunction.h"

namespace WebCore {

class SVGAnimationAdditiveFunction : public SVGAnimationFunction {
public:
    SVGAnimationAdditiveFunction(AnimationMode animationMode, CalcMode calcMode, bool isAccumulated, bool isAdditive)
        : SVGAnimationFunction(animationMode)
        , m_calcMode(calcMode)
        , m_isAccumulated(isAccumulated)
        , m_isAdditive(isAdditive)
    {
    }

    void setFromAndByValues(SVGElement& targetElement, const String& from, const String& by) override
    {
        setFromAndToValues(targetElement, from, by);
        addFromAndToValues(targetElement);
    }

    void setToAtEndOfDurationValue(const String&) override
    {
        ASSERT_NOT_REACHED();
    }

protected:
    float animate(float progress, unsigned repeatCount, float from, float to, float toAtEndOfDuration, float animated)
    {
        float number;
        if (m_calcMode == CalcMode::Discrete)
            number = progress < 0.5 ? from : to;
        else
            number = (to - from) * progress + from;

        if (m_isAccumulated && repeatCount)
            number += toAtEndOfDuration * repeatCount;

        if (m_isAdditive && m_animationMode != AnimationMode::To)
            number += animated;

        return number;
    }

    CalcMode m_calcMode;
    bool m_isAccumulated;
    bool m_isAdditive;
};

} // namespace WebCore
