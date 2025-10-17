/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 3, 2024.
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

#include "LegacyRenderSVGResourceContainer.h"

namespace WebCore {

class AffineTransform;
class RenderObject;
class SVGMarkerElement;

class LegacyRenderSVGResourceMarker final : public LegacyRenderSVGResourceContainer {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(LegacyRenderSVGResourceMarker);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(LegacyRenderSVGResourceMarker);
public:
    LegacyRenderSVGResourceMarker(SVGMarkerElement&, RenderStyle&&);
    virtual ~LegacyRenderSVGResourceMarker();

    inline SVGMarkerElement& markerElement() const;

    void removeAllClientsFromCache() override { }
    void removeClientFromCache(RenderElement&) override { }

    void draw(PaintInfo&, const AffineTransform&);

    // Calculates marker boundaries, mapped to the target element's coordinate space
    FloatRect markerBoundaries(RepaintRectCalculation, const AffineTransform& markerTransformation) const;

    void applyViewportClip(PaintInfo&) override;
    void layout() override;
    void calcViewport() override;

    const AffineTransform& localToParentTransform() const override;
    AffineTransform markerTransformation(const FloatPoint& origin, float angle, float strokeWidth) const;

    OptionSet<ApplyResult> applyResource(RenderElement&, const RenderStyle&, GraphicsContext*&, OptionSet<RenderSVGResourceMode>) override { return { }; }
    FloatRect resourceBoundingBox(const RenderObject&, RepaintRectCalculation) override { return FloatRect(); }

    FloatPoint referencePoint() const;
    std::optional<float> angle() const;
    inline SVGMarkerUnitsType markerUnits() const;

    RenderSVGResourceType resourceType() const override { return MarkerResourceType; }

private:
    void element() const = delete;

    ASCIILiteral renderName() const override { return "RenderSVGResourceMarker"_s; }

    // Generates a transformation matrix usable to render marker content. Handles scaling the marker content
    // acording to SVGs markerUnits="strokeWidth" concept, when a strokeWidth value != -1 is passed in.
    AffineTransform markerContentTransformation(const AffineTransform& contentTransformation, const FloatPoint& origin, float strokeWidth = -1) const;

    AffineTransform viewportTransform() const;

    mutable AffineTransform m_localToParentTransform;
    FloatRect m_viewport;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::LegacyRenderSVGResourceMarker)
static bool isType(const WebCore::RenderObject& renderer) { return renderer.isLegacyRenderSVGResourceMarker(); }
static bool isType(const WebCore::LegacyRenderSVGResource& resource) { return resource.resourceType() == WebCore::MarkerResourceType; }
SPECIALIZE_TYPE_TRAITS_END()
