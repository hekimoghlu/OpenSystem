/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 4, 2023.
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

#include "RenderSVGContainer.h"

namespace WebCore {
    
class SVGElement;

// This class is for containers which are never drawn, but do need to support style
// <defs>, <linearGradient>, <radialGradient> are all good examples
class RenderSVGHiddenContainer : public RenderSVGContainer {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderSVGHiddenContainer);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderSVGHiddenContainer);
public:
    RenderSVGHiddenContainer(Type, SVGElement&, RenderStyle&&, OptionSet<SVGModelObjectFlag> = { });
    virtual ~RenderSVGHiddenContainer();

protected:
    void layout() override;

private:
    ASCIILiteral renderName() const override { return "RenderSVGHiddenContainer"_s; }

    void paint(PaintInfo&, const LayoutPoint&) final { }

    LayoutRect clippedOverflowRect(const RenderLayerModelObject*, VisibleRectContext) const final { return { }; }
    std::optional<RepaintRects> computeVisibleRectsInContainer(const RepaintRects& rects, const RenderLayerModelObject*, VisibleRectContext) const final { return rects; }

    void boundingRects(Vector<LayoutRect>&, const LayoutPoint&) const final { }
    void absoluteQuads(Vector<FloatQuad>&, bool*) const final { }
    void addFocusRingRects(Vector<LayoutRect>&, const LayoutPoint&, const RenderLayerModelObject* = nullptr) const final { }

protected:
    bool nodeAtPoint(const HitTestRequest&, HitTestResult&, const HitTestLocation&, const LayoutPoint&, HitTestAction) final { return false; }
    void applyTransform(TransformationMatrix&, const RenderStyle&, const FloatRect&, OptionSet<RenderStyle::TransformOperationOption>) const override { }
    void updateFromStyle() override { }
    bool needsHasSVGTransformFlags() const override { return false; }
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderSVGHiddenContainer, isRenderSVGHiddenContainer())
