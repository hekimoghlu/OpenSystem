/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 9, 2022.
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

#include "LegacyRenderSVGModelObject.h"

namespace WebCore {

class SVGElement;

class LegacyRenderSVGContainer : public LegacyRenderSVGModelObject {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(LegacyRenderSVGContainer);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(LegacyRenderSVGContainer);
public:
    virtual ~LegacyRenderSVGContainer();

    void paint(PaintInfo&, const LayoutPoint&) override;
    void setNeedsBoundariesUpdate() final { m_needsBoundariesUpdate = true; }
    virtual bool didTransformToRootUpdate() { return false; }
    bool isObjectBoundingBoxValid() const { return m_objectBoundingBoxValid; }
    bool isRepaintSuspendedForChildren() const { return m_repaintIsSuspendedForChildrenDuringLayout; }

protected:
    LegacyRenderSVGContainer(Type, SVGElement&, RenderStyle&&, OptionSet<SVGModelObjectFlag> = { });

    ASCIILiteral renderName() const override { return "RenderSVGContainer"_s; }

    bool canHaveChildren() const final { return true; }

    void layout() override;

    void addFocusRingRects(Vector<LayoutRect>&, const LayoutPoint& additionalOffset, const RenderLayerModelObject* paintContainer = 0) const final;

    FloatRect objectBoundingBox() const final { return m_objectBoundingBox; }
    FloatRect strokeBoundingBox() const final;
    FloatRect repaintRectInLocalCoordinates(RepaintRectCalculation = RepaintRectCalculation::Fast) const final;

    bool nodeAtFloatPoint(const HitTestRequest&, HitTestResult&, const FloatPoint& pointInParent, HitTestAction) override;

    // Allow LegacyRenderSVGTransformableContainer to hook in at the right time in layout()
    virtual bool calculateLocalTransform() { return false; }

    // Allow RenderSVGViewportContainer to hook in at the right times in layout(), paint() and nodeAtFloatPoint()
    virtual void calcViewport() { }
    virtual void applyViewportClip(PaintInfo&) { }
    virtual bool pointIsInsideViewportClip(const FloatPoint& /*pointInParent*/) { return true; }

    virtual void determineIfLayoutSizeChanged() { }

    bool selfWillPaint();
    void updateCachedBoundaries();

private:
    FloatRect m_objectBoundingBox;
    mutable Markable<FloatRect, FloatRect::MarkableTraits> m_strokeBoundingBox;
    FloatRect m_repaintBoundingBox;
    mutable Markable<FloatRect, FloatRect::MarkableTraits> m_accurateRepaintBoundingBox;

    bool m_objectBoundingBoxValid { false };
    bool m_needsBoundariesUpdate { true };
    bool m_repaintIsSuspendedForChildrenDuringLayout { false };
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(LegacyRenderSVGContainer, isLegacyRenderSVGContainer())
