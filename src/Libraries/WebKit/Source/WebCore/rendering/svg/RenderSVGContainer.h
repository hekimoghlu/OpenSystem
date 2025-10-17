/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 10, 2025.
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

#include "RenderSVGModelObject.h"
#include "SVGBoundingBoxComputation.h"

namespace WebCore {

class SVGElement;

class RenderSVGContainer : public RenderSVGModelObject {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderSVGContainer);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderSVGContainer);
public:
    virtual ~RenderSVGContainer();

    void paint(PaintInfo&, const LayoutPoint&) override;

    bool isObjectBoundingBoxValid() const { return m_objectBoundingBoxValid; }
    bool isLayoutSizeChanged() const { return m_isLayoutSizeChanged; }
    bool didTransformToRootUpdate() const { return m_didTransformToRootUpdate; }

    FloatRect objectBoundingBox() const final { return m_objectBoundingBox; }
    FloatRect objectBoundingBoxWithoutTransformations() const final { return m_objectBoundingBoxWithoutTransformations; }
    FloatRect strokeBoundingBox() const final;
    FloatRect repaintRectInLocalCoordinates(RepaintRectCalculation = RepaintRectCalculation::Fast) const final { return SVGBoundingBoxComputation::computeRepaintBoundingBox(*this); }

protected:
    RenderSVGContainer(Type, Document&, RenderStyle&&, OptionSet<SVGModelObjectFlag> = { });
    RenderSVGContainer(Type, SVGElement&, RenderStyle&&, OptionSet<SVGModelObjectFlag> = { });

    ASCIILiteral renderName() const override { return "RenderSVGContainer"_s; }
    bool canHaveChildren() const final { return true; }

    void layout() override;

    virtual void layoutChildren();
    virtual bool pointIsInsideViewportClip(const FloatPoint&) { return true; }
    virtual bool updateLayoutSizeIfNeeded() { return false; }
    virtual std::optional<FloatRect> overridenObjectBoundingBoxWithoutTransformations() const { return std::nullopt; }
    bool nodeAtPoint(const HitTestRequest&, HitTestResult&, const HitTestLocation& locationInContainer, const LayoutPoint& accumulatedOffset, HitTestAction) override;

    bool m_objectBoundingBoxValid { false };
    bool m_isLayoutSizeChanged { false };
    bool m_didTransformToRootUpdate { false };
    FloatRect m_objectBoundingBox;
    FloatRect m_objectBoundingBoxWithoutTransformations;
    mutable Markable<FloatRect, FloatRect::MarkableTraits> m_strokeBoundingBox;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderSVGContainer, isRenderSVGContainer())

