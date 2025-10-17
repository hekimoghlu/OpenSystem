/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 12, 2023.
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

#include "SVGPointerMemberAccessor.h"

namespace WebCore {

template<typename OwnerType, typename AnimatedPropertyType>
class SVGAnimatedPropertyAccessor : public SVGPointerMemberAccessor<OwnerType, AnimatedPropertyType> {
    using Base = SVGPointerMemberAccessor<OwnerType, AnimatedPropertyType>;

public:
    using Base::Base;
    using Base::singleton;
    using Base::property;
    using AnimatedProperty = AnimatedPropertyType;

    bool matches(const OwnerType& owner, const SVGAnimatedProperty& animatedProperty) const override
    {
        return property(owner).ptr() == &animatedProperty;
    }

private:
    bool isAnimatedProperty() const override { return true; }
};

}
