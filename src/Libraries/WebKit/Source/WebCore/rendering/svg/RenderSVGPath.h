/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 22, 2024.
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

#include "RenderSVGShape.h"

namespace WebCore {

class RenderSVGPath final : public RenderSVGShape {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderSVGPath);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderSVGPath);
public:
    RenderSVGPath(SVGGraphicsElement&, RenderStyle&&);
    virtual ~RenderSVGPath();

    FloatRect computeMarkerBoundingBox(const SVGBoundingBoxComputation::DecorationOptions&) const;

    void updateMarkerPositions();

private:
    ASCIILiteral renderName() const override { return "RenderSVGPath"_s; }

    void updateShapeFromElement() override;
    FloatRect adjustStrokeBoundingBoxForZeroLengthLinecaps(RepaintRectCalculation, FloatRect strokeBoundingBox) const override;

    void strokeShape(GraphicsContext&) const override;
    bool shapeDependentStrokeContains(const FloatPoint&, PointCoordinateSpace = GlobalCoordinateSpace) override;

    void styleDidChange(StyleDifference, const RenderStyle*) final;

    bool shouldStrokeZeroLengthSubpath() const;
    Path* zeroLengthLinecapPath(const FloatPoint&) const;
    FloatRect zeroLengthSubpathRect(const FloatPoint&, float) const;
    void updateZeroLengthSubpaths();
    void strokeZeroLengthSubpaths(GraphicsContext&) const;

    bool shouldGenerateMarkerPositions() const;
    void drawMarkers(PaintInfo&) override;

    bool isRenderingDisabled() const override;

    Vector<FloatPoint> m_zeroLengthLinecapLocations;
    Vector<MarkerPosition> m_markerPositions;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderSVGPath, isRenderSVGPath())
