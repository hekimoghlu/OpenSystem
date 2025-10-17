/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 28, 2025.
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

#include "SVGAnimationAdditiveFunction.h"

namespace WebCore {

template<typename ListType>
class SVGAnimationAdditiveListFunction : public SVGAnimationAdditiveFunction {
public:
    template<typename... Arguments>
    SVGAnimationAdditiveListFunction(AnimationMode animationMode, CalcMode calcMode, bool isAccumulated, bool isAdditive, Arguments&&... arguments)
        : SVGAnimationAdditiveFunction(animationMode, calcMode, isAccumulated, isAdditive)
        , m_from(ListType::create(std::forward<Arguments>(arguments)...))
        , m_to(ListType::create(std::forward<Arguments>(arguments)...))
        , m_toAtEndOfDuration(ListType::create(std::forward<Arguments>(arguments)...))
    {
    }

protected:
    const Ref<ListType>& toAtEndOfDuration() const { return !m_toAtEndOfDuration->isEmpty() ? m_toAtEndOfDuration : m_to; }

    bool adjustAnimatedList(AnimationMode animationMode, float percentage, RefPtr<ListType>& animated, bool resizeAnimatedIfNeeded = true)
    {
        if (!m_to->numberOfItems())
            return false;

        if (m_from->numberOfItems() && m_from->size() != m_to->size()) {
            if (percentage >= 0.5)
                *animated = m_to;
            else if (animationMode != AnimationMode::To)
                *animated = m_from;
            return false;
        }

        if (resizeAnimatedIfNeeded && animated->size() < m_to->size())
            animated->resize(m_to->size());
        return true;
    }

    Ref<ListType> m_from;
    Ref<ListType> m_to;
    Ref<ListType> m_toAtEndOfDuration;
};

}
