/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 14, 2022.
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
#include "RenderSVGRoot.h"

namespace WebCore {

class SVGSVGElement;

class RenderSVGViewportContainer final : public RenderSVGContainer {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderSVGViewportContainer);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderSVGViewportContainer);
public:
    RenderSVGViewportContainer(RenderSVGRoot&, RenderStyle&&);
    RenderSVGViewportContainer(SVGSVGElement&, RenderStyle&&);
    virtual ~RenderSVGViewportContainer();

    SVGSVGElement& svgSVGElement() const;
    Ref<SVGSVGElement> protectedSVGSVGElement() const;
    FloatRect viewport() const { return { { }, viewportSize() }; }
    FloatSize viewportSize() const { return m_viewport.size(); }

    void updateFromStyle() final;

private:
    ASCIILiteral renderName() const final { return "RenderSVGViewportContainer"_s; }

    void element() const = delete;

    bool isOutermostSVGViewportContainer() const { return isAnonymous(); }
    bool updateLayoutSizeIfNeeded() final;
    std::optional<FloatRect> overridenObjectBoundingBoxWithoutTransformations() const final { return std::make_optional(viewport()); }

    FloatPoint computeViewportLocation() const;
    FloatSize computeViewportSize() const;

    void applyTransform(TransformationMatrix&, const RenderStyle&, const FloatRect& boundingBox, OptionSet<RenderStyle::TransformOperationOption>) const final;
    LayoutRect overflowClipRect(const LayoutPoint& location, OverlayScrollbarSizeRelevancy = OverlayScrollbarSizeRelevancy::IgnoreOverlayScrollbarSize, PaintPhase = PaintPhase::BlockBackground) const final;
    void updateLayerTransform() final;
    bool needsHasSVGTransformFlags() const final;

    AffineTransform m_supplementalLayerTransform;
    FloatRect m_viewport;
    SingleThreadWeakPtr<RenderSVGRoot> m_owningSVGRoot;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderSVGViewportContainer, isRenderSVGViewportContainer())

