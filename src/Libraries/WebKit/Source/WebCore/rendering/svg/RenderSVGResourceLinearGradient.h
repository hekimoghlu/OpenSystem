/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 19, 2022.
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

#include "AffineTransform.h"
#include "LinearGradientAttributes.h"
#include "RenderSVGResourceGradient.h"
#include "SVGGradientElement.h"
#include "SVGUnitTypes.h"

namespace WebCore {

class SVGLinearGradientElement;

class RenderSVGResourceLinearGradient final : public RenderSVGResourceGradient {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderSVGResourceLinearGradient);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderSVGResourceLinearGradient);
public:
    RenderSVGResourceLinearGradient(SVGLinearGradientElement&, RenderStyle&&);
    virtual ~RenderSVGResourceLinearGradient();

    inline SVGLinearGradientElement& linearGradientElement() const;
    inline Ref<SVGLinearGradientElement> protectedLinearGradientElement() const;

    SVGUnitTypes::SVGUnitType gradientUnits() const final { return m_attributes ? m_attributes.value().gradientUnits() : SVGUnitTypes::SVG_UNIT_TYPE_UNKNOWN; }
    AffineTransform gradientTransform() const final { return m_attributes ? m_attributes.value().gradientTransform() : identity; }

    void invalidateGradient() final
    {
        m_gradient = nullptr;
        m_attributes = std::nullopt;
        repaintAllClients();
    }

private:
    void collectGradientAttributesIfNeeded() final;
    RefPtr<Gradient> createGradient(const RenderStyle&) final;

    void element() const = delete;
    ASCIILiteral renderName() const final { return "RenderSVGResourceLinearGradient"_s; }

    std::optional<LinearGradientAttributes> m_attributes;
};

}

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderSVGResourceLinearGradient, isRenderSVGResourceLinearGradient())
