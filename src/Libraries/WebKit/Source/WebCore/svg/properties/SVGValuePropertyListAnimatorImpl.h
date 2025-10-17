/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 21, 2025.
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
#include "SVGValuePropertyListAnimator.h"

namespace WebCore {

class SVGLengthListAnimator final : public SVGValuePropertyListAnimator<SVGLengthList, SVGAnimationLengthListFunction> {
    using Base = SVGValuePropertyListAnimator<SVGLengthList, SVGAnimationLengthListFunction>;
    using Base::Base;
    using Base::m_attributeName;
    using Base::m_list;

public:
    static auto create(const QualifiedName& attributeName, Ref<SVGProperty>&& property, AnimationMode animationMode, CalcMode calcMode, bool isAccumulated, bool isAdditive)
    {
        return adoptRef(*new SVGLengthListAnimator(attributeName, WTFMove(property), animationMode, calcMode, isAccumulated, isAdditive, SVGLengthMode::Other));
    }

    void start(SVGElement& targetElement) final
    {
        String baseValue = computeCSSPropertyValue(targetElement, cssPropertyID(m_attributeName.localName()));
        if (!m_list->parse(baseValue))
            m_list->clearItems();
    }
};

} // namespace WebCore
