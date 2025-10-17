/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 23, 2021.
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

#include "LegacyRenderSVGContainer.h"

namespace WebCore {

// This is used for non-root <svg> elements and <marker> elements, neither of which are SVGTransformable
// thus we inherit from LegacyRenderSVGContainer instead of LegacyRenderSVGTransformableContainer
class LegacyRenderSVGViewportContainer final : public LegacyRenderSVGContainer {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(LegacyRenderSVGViewportContainer);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(LegacyRenderSVGViewportContainer);
public:
    LegacyRenderSVGViewportContainer(SVGSVGElement&, RenderStyle&&);
    virtual ~LegacyRenderSVGViewportContainer();

    SVGSVGElement& svgSVGElement() const;

    FloatRect viewport() const { return m_viewport; }

    bool isLayoutSizeChanged() const { return m_isLayoutSizeChanged; }
    bool didTransformToRootUpdate() override { return m_didTransformToRootUpdate; }

    void determineIfLayoutSizeChanged() override;
    void setNeedsTransformUpdate() override { m_needsTransformUpdate = true; }

    void paint(PaintInfo&, const LayoutPoint&) override;

private:
    void element() const = delete;

    ASCIILiteral renderName() const override { return "RenderSVGViewportContainer"_s; }

    AffineTransform viewportTransform() const;
    const AffineTransform& localToParentTransform() const override { return m_localToParentTransform; }

    void calcViewport() override;
    bool calculateLocalTransform() override;

    void applyViewportClip(PaintInfo&) override;
    bool pointIsInsideViewportClip(const FloatPoint& pointInParent) override;

    bool m_didTransformToRootUpdate : 1;
    bool m_isLayoutSizeChanged : 1;
    bool m_needsTransformUpdate : 1;

    FloatRect m_viewport;
    mutable AffineTransform m_localToParentTransform;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(LegacyRenderSVGViewportContainer, isLegacyRenderSVGViewportContainer())
