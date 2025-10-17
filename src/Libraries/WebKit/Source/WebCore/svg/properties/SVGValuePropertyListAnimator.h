/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 9, 2023.
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

#include "SVGPropertyAnimator.h"

namespace WebCore {

template<typename ListType, typename AnimationFunction>
class SVGValuePropertyListAnimator : public SVGPropertyAnimator<AnimationFunction> {
    WTF_MAKE_TZONE_ALLOCATED_TEMPLATE(SVGValuePropertyListAnimator);
    using Base = SVGPropertyAnimator<AnimationFunction>;
    using Base::Base;
    using Base::applyAnimatedStylePropertyChange;
    using Base::m_function;

public:
    template<typename... Arguments>
    SVGValuePropertyListAnimator(const QualifiedName& attributeName, Ref<SVGProperty>&& property, Arguments&&... arguments)
        : Base(attributeName, std::forward<Arguments>(arguments)...)
        , m_list(static_reference_cast<ListType>(WTFMove(property)))
    {
    }

    void animate(SVGElement& targetElement, float progress, unsigned repeatCount) override
    {
        m_function.animate(targetElement, progress, repeatCount, m_list);
    }

    void apply(SVGElement& targetElement) override
    {
        applyAnimatedStylePropertyChange(targetElement, m_list->valueAsString());
    }

protected:
    using Base::computeCSSPropertyValue;
    using Base::m_attributeName;

    RefPtr<ListType> m_list;
};

#define TZONE_TEMPLATE_PARAMS template<typename ListType, typename AnimationFunction>
#define TZONE_TYPE SVGValuePropertyListAnimator<ListType, AnimationFunction>

WTF_MAKE_TZONE_ALLOCATED_TEMPLATE_IMPL_WITH_MULTIPLE_OR_SPECIALIZED_PARAMETERS();

#undef TZONE_TEMPLATE_PARAMS
#undef TZONE_TYPE

} // namespace WebCore
