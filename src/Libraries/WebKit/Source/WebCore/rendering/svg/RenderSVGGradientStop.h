/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 6, 2025.
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

namespace WebCore {
    
class SVGGradientElement;
class SVGStopElement;

// This class exists mostly so we can hear about gradient stop style changes
class RenderSVGGradientStop final : public RenderElement {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderSVGGradientStop);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderSVGGradientStop);
public:
    RenderSVGGradientStop(SVGStopElement&, RenderStyle&&);
    virtual ~RenderSVGGradientStop();

    inline SVGStopElement& element() const;

private:
    void styleDidChange(StyleDifference, const RenderStyle* oldStyle) override;

    void layout() override;

    // These overrides are needed to prevent ASSERTs on <svg><stop /></svg>
    // RenderObject's default implementations ASSERT_NOT_REACHED()
    // https://bugs.webkit.org/show_bug.cgi?id=20400
    RepaintRects localRectsForRepaint(RepaintOutlineBounds) const override { return { }; }
    FloatRect objectBoundingBox() const override { return { }; }
    FloatRect strokeBoundingBox() const override { return { }; }
    FloatRect repaintRectInLocalCoordinates(RepaintRectCalculation) const override { return { }; }
    bool nodeAtFloatPoint(const HitTestRequest&, HitTestResult&, const FloatPoint&, HitTestAction) override { return false; }

    ASCIILiteral renderName() const override { return "RenderSVGGradientStop"_s; }

    bool canHaveChildren() const override { return false; }
    void paint(PaintInfo&, const LayoutPoint&) override { }

    SVGGradientElement* gradientElement();
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderSVGGradientStop, isRenderSVGGradientStop())
