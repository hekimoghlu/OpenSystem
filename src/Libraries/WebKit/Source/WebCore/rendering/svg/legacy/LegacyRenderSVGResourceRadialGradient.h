/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 23, 2023.
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
#include "RadialGradientAttributes.h"

namespace WebCore {

class SVGRadialGradientElement;

class LegacyRenderSVGResourceRadialGradient final : public LegacyRenderSVGResourceGradient {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(LegacyRenderSVGResourceRadialGradient);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(LegacyRenderSVGResourceRadialGradient);
public:
    LegacyRenderSVGResourceRadialGradient(SVGRadialGradientElement&, RenderStyle&&);
    virtual ~LegacyRenderSVGResourceRadialGradient();

    inline SVGRadialGradientElement& radialGradientElement() const;
    inline Ref<SVGRadialGradientElement> protectedRadialGradientElement() const;

    FloatPoint centerPoint(const RadialGradientAttributes&) const;
    FloatPoint focalPoint(const RadialGradientAttributes&) const;
    float radius(const RadialGradientAttributes&) const;
    float focalRadius(const RadialGradientAttributes&) const;

private:
    RenderSVGResourceType resourceType() const final { return RadialGradientResourceType; }

    SVGUnitTypes::SVGUnitType gradientUnits() const final { return m_attributes.gradientUnits(); }
    AffineTransform gradientTransform() const final { return m_attributes.gradientTransform(); }
    Ref<Gradient> buildGradient(const RenderStyle&) const final;

    void gradientElement() const = delete;

    ASCIILiteral renderName() const final { return "RenderSVGResourceRadialGradient"_s; }
    bool collectGradientAttributes() final;

    RadialGradientAttributes m_attributes;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_LEGACY_RENDER_SVG_RESOURCE(LegacyRenderSVGResourceRadialGradient, RadialGradientResourceType)
