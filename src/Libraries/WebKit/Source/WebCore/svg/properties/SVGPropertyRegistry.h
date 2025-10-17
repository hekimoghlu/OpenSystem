/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 13, 2022.
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

#include "SVGAttributeAnimator.h"

namespace WebCore {

class SVGAnimatedProperty;

class SVGPropertyRegistry {
public:
    SVGPropertyRegistry() = default;
    virtual ~SVGPropertyRegistry() = default;

    virtual void detachAllProperties() const = 0;
    virtual QualifiedName propertyAttributeName(const SVGProperty&) const = 0;
    virtual QualifiedName animatedPropertyAttributeName(const SVGAnimatedProperty&) const = 0;
    virtual void setAnimatedPropertyDirty(const QualifiedName&, SVGAnimatedProperty&) const = 0;
    virtual std::optional<String> synchronize(const QualifiedName&) const = 0;
    virtual UncheckedKeyHashMap<QualifiedName, String> synchronizeAllAttributes() const = 0;

    virtual bool isAnimatedPropertyAttribute(const QualifiedName&) const = 0;
    virtual bool isAnimatedStylePropertyAttribute(const QualifiedName&) const = 0;
    virtual RefPtr<SVGAttributeAnimator> createAnimator(const QualifiedName&, AnimationMode, CalcMode, bool isAccumulated, bool isAdditive) const = 0;
    virtual void appendAnimatedInstance(const QualifiedName& attributeName, SVGAttributeAnimator&) const = 0;
};

}
