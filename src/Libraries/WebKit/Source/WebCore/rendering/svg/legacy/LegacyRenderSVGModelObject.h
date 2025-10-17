/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 3, 2024.
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

#include "RenderElement.h"
#include "SVGElement.h"
#include "SVGRenderSupport.h"

namespace WebCore {

// Most renderers in the SVG rendering tree will inherit from this class
// but not all. (e.g. LegacyRenderSVGForeignObject, RenderSVGBlock) thus methods
// required by SVG renders need to be declared on RenderObject, but shared
// logic can go in this class or in SVGRenderSupport.

class SVGElement;

class LegacyRenderSVGModelObject : public RenderElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(LegacyRenderSVGModelObject);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(LegacyRenderSVGModelObject);
public:
    virtual ~LegacyRenderSVGModelObject();

    LayoutRect clippedOverflowRect(const RenderLayerModelObject* repaintContainer, VisibleRectContext) const override;
    RepaintRects rectsForRepaintingAfterLayout(const RenderLayerModelObject* repaintContainer, RepaintOutlineBounds) const override;

    std::optional<FloatRect> computeFloatVisibleRectInContainer(const FloatRect&, const RenderLayerModelObject* container, VisibleRectContext) const final;
    LayoutRect outlineBoundsForRepaint(const RenderLayerModelObject* repaintContainer, const RenderGeometryMap* = nullptr) const final;

    void boundingRects(Vector<LayoutRect>&, const LayoutPoint& accumulatedOffset) const final;
    void absoluteQuads(Vector<FloatQuad>&, bool* wasFixed) const override;

    void mapLocalToContainer(const RenderLayerModelObject* ancestorContainer, TransformState&, OptionSet<MapCoordinatesMode>, bool* wasFixed) const final;
    const RenderObject* pushMappingToContainer(const RenderLayerModelObject* ancestorToStopAt, RenderGeometryMap&) const final;
    void styleDidChange(StyleDifference, const RenderStyle* oldStyle) override;

    static bool checkIntersection(RenderElement*, const FloatRect&);
    static bool checkEnclosure(RenderElement*, const FloatRect&);

    SVGElement& element() const { return downcast<SVGElement>(nodeForNonAnonymous()); }
    Ref<SVGElement> protectedElement() const;

protected:
    LegacyRenderSVGModelObject(Type, SVGElement&, RenderStyle&&, OptionSet<SVGModelObjectFlag> = { });

    void willBeDestroyed() override;

private:
    // This method should never be called, SVG uses a different nodeAtPoint method
    bool nodeAtPoint(const HitTestRequest&, HitTestResult&, const HitTestLocation& locationInContainer, const LayoutPoint& accumulatedOffset, HitTestAction) override;
    void absoluteFocusRingQuads(Vector<FloatQuad>&) final;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(LegacyRenderSVGModelObject, isLegacyRenderSVGModelObject())
