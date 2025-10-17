/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 7, 2023.
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

namespace WebCore {

template<typename ValueType>
class SVGAnimationDiscreteFunction : public SVGAnimationFunction {
public:
    SVGAnimationDiscreteFunction(AnimationMode animationMode, CalcMode, bool, bool)
        : SVGAnimationFunction(animationMode)
    {
    }

    bool isDiscrete() const override { return true; }

    void setToAtEndOfDurationValue(const String&) override
    {
        ASSERT_NOT_REACHED();
    }

    void setFromAndByValues(SVGElement&, const String&, const String&) override
    {
        ASSERT_NOT_REACHED();
    }

    void animate(SVGElement&, float progress, unsigned, ValueType& animated)
    {
        if ((m_animationMode == AnimationMode::FromTo && progress > 0.5) || m_animationMode == AnimationMode::To || progress == 1)
            animated = m_to;
        else
            animated = m_from;
    }

protected:
    ValueType m_from;
    ValueType m_to;
};

} // namespace WebCore
