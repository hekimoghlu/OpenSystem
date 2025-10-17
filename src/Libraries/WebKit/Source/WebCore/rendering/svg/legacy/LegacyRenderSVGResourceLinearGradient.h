/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 24, 2025.
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

#include "LegacyRenderSVGResourceGradient.h"
#include "LinearGradientAttributes.h"

namespace WebCore {

class SVGLinearGradientElement;

class LegacyRenderSVGResourceLinearGradient final : public LegacyRenderSVGResourceGradient {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(LegacyRenderSVGResourceLinearGradient);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(LegacyRenderSVGResourceLinearGradient);
public:
    LegacyRenderSVGResourceLinearGradient(SVGLinearGradientElement&, RenderStyle&&);
    virtual ~LegacyRenderSVGResourceLinearGradient();

    inline SVGLinearGradientElement& linearGradientElement() const;
    inline Ref<SVGLinearGradientElement> protectedLinearGradientElement() const;

    FloatPoint startPoint(const LinearGradientAttributes&) const;
    FloatPoint endPoint(const LinearGradientAttributes&) const;

private:
    RenderSVGResourceType resourceType() const final { return LinearGradientResourceType; }

    SVGUnitTypes::SVGUnitType gradientUnits() const final { return m_attributes.gradientUnits(); }
    AffineTransform gradientTransform() const final { return m_attributes.gradientTransform(); }
    bool collectGradientAttributes() final;
    Ref<Gradient> buildGradient(const RenderStyle&) const final;

    void gradientElement() const = delete;

    ASCIILiteral renderName() const final { return "RenderSVGResourceLinearGradient"_s; }

    LinearGradientAttributes m_attributes;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_LEGACY_RENDER_SVG_RESOURCE(LegacyRenderSVGResourceLinearGradient, LinearGradientResourceType)
