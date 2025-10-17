/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 28, 2025.
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

#include "LegacyRenderSVGShape.h"

namespace WebCore {

class LegacyRenderSVGPath final : public LegacyRenderSVGShape {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(LegacyRenderSVGPath);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(LegacyRenderSVGPath);
public:
    LegacyRenderSVGPath(SVGGraphicsElement&, RenderStyle&&);
    virtual ~LegacyRenderSVGPath();

    void drawMarkers(PaintInfo&) final;
    FloatRect adjustStrokeBoundingBoxForMarkersAndZeroLengthLinecaps(RepaintRectCalculation, FloatRect strokeBoundingBox) const override;

private:
    ASCIILiteral renderName() const override { return "RenderSVGPath"_s; }

    void updateShapeFromElement() override;

    void strokeShape(GraphicsContext&) const override;
    bool shapeDependentStrokeContains(const FloatPoint&, PointCoordinateSpace = GlobalCoordinateSpace) override;

    void styleDidChange(StyleDifference, const RenderStyle*) final;

    bool shouldStrokeZeroLengthSubpath() const;
    Path* zeroLengthLinecapPath(const FloatPoint&) const;
    FloatRect zeroLengthSubpathRect(const FloatPoint&, float) const;
    void updateZeroLengthSubpaths();
    void strokeZeroLengthSubpaths(GraphicsContext&) const;

    bool shouldGenerateMarkerPositions() const;
    void processMarkerPositions();
    FloatRect markerRect(RepaintRectCalculation, float strokeWidth) const;

    bool isRenderingDisabled() const override;

    Vector<FloatPoint> m_zeroLengthLinecapLocations;
    Vector<MarkerPosition> m_markerPositions;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(LegacyRenderSVGPath, isLegacyRenderSVGPath())
