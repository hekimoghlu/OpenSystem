/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 30, 2022.
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
#include "SVGMarkerTypes.h"

namespace WebCore {

class GraphicsContext;
class SVGMarkerElement;

class RenderSVGResourceMarker final : public RenderSVGResourceContainer {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderSVGResourceMarker);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderSVGResourceMarker);
public:
    RenderSVGResourceMarker(SVGMarkerElement&, RenderStyle&&);
    virtual ~RenderSVGResourceMarker();

    inline bool hasReverseStart() const;

    void invalidateMarker();

    // Calculates marker boundaries, mapped to the target element's coordinate space
    FloatRect computeMarkerBoundingBox(const SVGBoundingBoxComputation::DecorationOptions&, const AffineTransform& markerTransformation) const;

    AffineTransform markerTransformation(const FloatPoint& origin, float angle, float strokeWidth) const;

private:
    ASCIILiteral renderName() const final { return "RenderSVGResourceMarker"_s; }

    inline SVGMarkerElement& markerElement() const;
    inline Ref<SVGMarkerElement> protectedMarkerElement() const;
    inline FloatPoint referencePoint() const;
    inline std::optional<float> angle() const;
    inline SVGMarkerUnitsType markerUnits() const;

    FloatRect viewport() const { return m_viewport; }
    FloatSize viewportSize() const { return m_viewport.size(); }

    void element() const = delete;
    bool updateLayoutSizeIfNeeded() final;
    std::optional<FloatRect> overridenObjectBoundingBoxWithoutTransformations() const final { return std::make_optional(viewport()); }

    FloatRect computeViewport() const;

    void applyTransform(TransformationMatrix&, const RenderStyle&, const FloatRect& boundingBox, OptionSet<RenderStyle::TransformOperationOption>) const final;
    LayoutRect overflowClipRect(const LayoutPoint& location, OverlayScrollbarSizeRelevancy = OverlayScrollbarSizeRelevancy::IgnoreOverlayScrollbarSize, PaintPhase = PaintPhase::BlockBackground) const final;
    void updateLayerTransform() final;
    bool needsHasSVGTransformFlags() const final { return true; }

    void layout() final;
    void updateFromStyle() final;

private:
    AffineTransform m_supplementalLayerTransform;
    FloatRect m_viewport;
};

}

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderSVGResourceMarker, isRenderSVGResourceMarker())
