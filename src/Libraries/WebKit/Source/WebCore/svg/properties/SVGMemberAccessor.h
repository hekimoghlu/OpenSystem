/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 23, 2025.
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

#include "QualifiedName.h"
#include "SVGAnimatedProperty.h"
#include "SVGAttributeAnimator.h"

namespace WebCore {

class SVGProperty;

template<typename OwnerType>
class SVGMemberAccessor {
    WTF_MAKE_TZONE_ALLOCATED_TEMPLATE(SVGMemberAccessor);
public:
    virtual ~SVGMemberAccessor() = default;

    virtual void detach(const OwnerType&) const { }
    virtual bool isAnimatedProperty() const { return false; }
    virtual bool isAnimatedLength() const { return false; }

    virtual bool matches(const OwnerType&, const SVGProperty&) const { return false; }
    virtual bool matches(const OwnerType&, const SVGAnimatedProperty&) const { return false; }
    virtual void setDirty(const OwnerType&, SVGAnimatedProperty& animatedProperty) const { animatedProperty.setDirty(); }
    virtual std::optional<String> synchronize(const OwnerType&) const { return std::nullopt; }

    virtual RefPtr<SVGAttributeAnimator> createAnimator(OwnerType&, const QualifiedName&, AnimationMode, CalcMode, bool, bool) const { return nullptr; }
    virtual void appendAnimatedInstance(OwnerType&, SVGAttributeAnimator&) const { }

protected:
    SVGMemberAccessor() = default;
};

WTF_MAKE_TZONE_ALLOCATED_TEMPLATE_IMPL(template<typename OwnerType>, SVGMemberAccessor<OwnerType>);

} // namespace WebCore
