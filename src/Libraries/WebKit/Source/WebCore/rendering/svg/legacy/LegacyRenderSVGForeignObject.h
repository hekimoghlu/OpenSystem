/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 29, 2023.
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

namespace WebCore {

class SVGForeignObjectElement;

class LegacyRenderSVGForeignObject final : public RenderSVGBlock {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(LegacyRenderSVGForeignObject);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(LegacyRenderSVGForeignObject);
public:
    LegacyRenderSVGForeignObject(SVGForeignObjectElement&, RenderStyle&&);
    virtual ~LegacyRenderSVGForeignObject();

    SVGForeignObjectElement& foreignObjectElement() const;

    void paint(PaintInfo&, const LayoutPoint&) override;

    bool requiresLayer() const override { return false; }
    void layout() override;

    FloatRect objectBoundingBox() const override { return FloatRect(FloatPoint(), m_viewport.size()); }
    FloatRect strokeBoundingBox() const override { return FloatRect(FloatPoint(), m_viewport.size()); }
    FloatRect repaintRectInLocalCoordinates(RepaintRectCalculation = RepaintRectCalculation::Fast) const override { return FloatRect(FloatPoint(), m_viewport.size()); }

    bool nodeAtFloatPoint(const HitTestRequest&, HitTestResult&, const FloatPoint& pointInParent, HitTestAction) override;

    void setNeedsTransformUpdate() override { m_needsTransformUpdate = true; }

private:
    void graphicsElement() const = delete;
    ASCIILiteral renderName() const override { return "RenderSVGForeignObject"_s; }

    void updateLogicalWidth() override;
    LogicalExtentComputedValues computeLogicalHeight(LayoutUnit logicalHeight, LayoutUnit logicalTop) const override;

    const AffineTransform& localToParentTransform() const override;
    AffineTransform localTransform() const override { return m_localTransform; }

    LayoutSize offsetFromContainer(RenderElement&, const LayoutPoint&, bool* offsetDependsOnPoint = nullptr) const override;

    AffineTransform m_localTransform;
    mutable AffineTransform m_localToParentTransform;
    FloatRect m_viewport;
    bool m_needsTransformUpdate { true };
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(LegacyRenderSVGForeignObject, isLegacyRenderSVGForeignObject())
