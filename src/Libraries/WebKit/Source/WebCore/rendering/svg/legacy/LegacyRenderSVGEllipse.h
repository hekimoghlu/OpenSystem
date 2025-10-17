/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 28, 2025.
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

class LegacyRenderSVGEllipse final : public LegacyRenderSVGShape {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(LegacyRenderSVGEllipse);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(LegacyRenderSVGEllipse);
public:
    LegacyRenderSVGEllipse(SVGGraphicsElement&, RenderStyle&&);
    virtual ~LegacyRenderSVGEllipse();

private:
    ASCIILiteral renderName() const override { return "RenderSVGEllipse"_s; }

    void updateShapeFromElement() override;
    bool isEmpty() const override { return hasPath() ? LegacyRenderSVGShape::isEmpty() : m_fillBoundingBox.isEmpty(); }
    bool isRenderingDisabled() const override;
    void fillShape(GraphicsContext&) const override;
    void strokeShape(GraphicsContext&) const override;
    bool shapeDependentStrokeContains(const FloatPoint&, PointCoordinateSpace = GlobalCoordinateSpace) override;
    bool shapeDependentFillContains(const FloatPoint&, const WindRule) const override;
    void calculateRadiiAndCenter();

private:
    bool canUseStrokeHitTestFastPath() const;

    FloatPoint m_center;
    FloatSize m_radii;
};

} // namespace WebCore
