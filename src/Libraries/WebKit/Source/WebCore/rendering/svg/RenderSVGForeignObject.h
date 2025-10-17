/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 24, 2022.
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
#include "FloatPoint.h"
#include "FloatRect.h"
#include "RenderSVGBlock.h"
#include "SVGBoundingBoxComputation.h"

namespace WebCore {

class SVGForeignObjectElement;

class RenderSVGForeignObject final : public RenderSVGBlock {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderSVGForeignObject);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderSVGForeignObject);
public:
    RenderSVGForeignObject(SVGForeignObjectElement&, RenderStyle&&);
    virtual ~RenderSVGForeignObject();

    SVGForeignObjectElement& foreignObjectElement() const;
    Ref<SVGForeignObjectElement> protectedForeignObjectElement() const;

    void paint(PaintInfo&, const LayoutPoint&) override;

    void layout() override;

    FloatRect objectBoundingBox() const final { return m_viewport; }
    FloatRect strokeBoundingBox() const final { return m_viewport; }
    FloatRect repaintRectInLocalCoordinates(RepaintRectCalculation = RepaintRectCalculation::Fast) const final { return SVGBoundingBoxComputation::computeRepaintBoundingBox(*this); }

private:
    void graphicsElement() const = delete;
    ASCIILiteral renderName() const override { return "RenderSVGForeignObject"_s; }

    void updateLogicalWidth() override;
    LogicalExtentComputedValues computeLogicalHeight(LayoutUnit logicalHeight, LayoutUnit logicalTop) const override;

    LayoutRect overflowClipRect(const LayoutPoint& location, OverlayScrollbarSizeRelevancy = OverlayScrollbarSizeRelevancy::IgnoreOverlayScrollbarSize, PaintPhase = PaintPhase::BlockBackground) const final;

    void updateFromStyle() final;

    // Enforce <fO> to carry a transform: <fO> should behave as absolutely positioned container
    // for CSS content. Thus it needs to become a rootPaintingLayer during paint() such that
    // fixed position content uses the <fO> as ancestor layer (when computing offsets from the container).
    bool needsHasSVGTransformFlags() const final { return true; }

    void applyTransform(TransformationMatrix&, const RenderStyle&, const FloatRect& boundingBox, OptionSet<RenderStyle::TransformOperationOption>) const final;

    FloatRect m_viewport;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderSVGForeignObject, isRenderSVGForeignObject())
