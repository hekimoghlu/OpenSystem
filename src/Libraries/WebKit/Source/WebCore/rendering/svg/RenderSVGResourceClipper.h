/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 6, 2023.
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

#include "RenderSVGResourceContainer.h"
#include "SVGUnitTypes.h"

namespace WebCore {

class GraphicsContext;
class SVGClipPathElement;
class SVGGraphicsElement;

class RenderSVGResourceClipper final : public RenderSVGResourceContainer {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderSVGResourceClipper);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderSVGResourceClipper);
public:
    RenderSVGResourceClipper(SVGClipPathElement&, RenderStyle&&);
    virtual ~RenderSVGResourceClipper();

    inline Ref<SVGClipPathElement> protectedClipPathElement() const;

    RefPtr<SVGGraphicsElement> shouldApplyPathClipping() const;
    void applyPathClipping(GraphicsContext&, const RenderLayerModelObject& targetRenderer, const FloatRect& objectBoundingBox, SVGGraphicsElement&);
    void applyMaskClipping(PaintInfo&, const RenderLayerModelObject& targetRenderer, const FloatRect& objectBoundingBox);

    FloatRect resourceBoundingBox(const RenderObject&, RepaintRectCalculation);

    bool hitTestClipContent(const FloatRect&, const LayoutPoint&);

    inline SVGUnitTypes::SVGUnitType clipPathUnits() const;

    void applyTransform(TransformationMatrix&, const RenderStyle&, const FloatRect& boundingBox, OptionSet<RenderStyle::TransformOperationOption>) const final;

private:
    void element() const = delete;

    bool needsHasSVGTransformFlags() const final;

    void updateFromStyle() final;

    ASCIILiteral renderName() const final { return "RenderSVGResourceClipper"_s; }

    void styleDidChange(StyleDifference, const RenderStyle* oldStyle) final;
};

}

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderSVGResourceClipper, isRenderSVGResourceClipper())

