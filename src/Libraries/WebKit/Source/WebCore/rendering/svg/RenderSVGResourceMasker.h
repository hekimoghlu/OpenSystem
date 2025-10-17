/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 20, 2024.
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

#include "ImageBuffer.h"
#include "RenderSVGResourceContainer.h"
#include "SVGUnitTypes.h"
#include <wtf/HashMap.h>

namespace WebCore {

class GraphicsContext;
class SVGMaskElement;

class RenderSVGResourceMasker final : public RenderSVGResourceContainer {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderSVGResourceMasker);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderSVGResourceMasker);
public:
    RenderSVGResourceMasker(SVGMaskElement&, RenderStyle&&);
    virtual ~RenderSVGResourceMasker();

    inline SVGMaskElement& maskElement() const;
    inline Ref<SVGMaskElement> protectedMaskElement() const;

    void applyMask(PaintInfo&, const RenderLayerModelObject& targetRenderer, const LayoutPoint& adjustedPaintOffset);

    FloatRect resourceBoundingBox(const RenderObject&, RepaintRectCalculation);

    inline SVGUnitTypes::SVGUnitType maskUnits() const;
    inline SVGUnitTypes::SVGUnitType maskContentUnits() const;

    void invalidateMask()
    {
        m_masker.clear();
    }

    void removeReferencingCSSClient(const RenderElement&) final;

    bool drawContentIntoContext(GraphicsContext&, const FloatRect& objectBoundingBox);
    bool drawContentIntoContext(GraphicsContext&, const FloatRect& destinationRect, const FloatRect& sourceRect, ImagePaintingOptions);

private:
    void element() const = delete;

    ASCIILiteral renderName() const final { return "RenderSVGResourceMasker"_s; }
    UncheckedKeyHashMap<SingleThreadWeakRef<const RenderLayerModelObject>, RefPtr<ImageBuffer>> m_masker;
};

}

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderSVGResourceMasker, isRenderSVGResourceMasker())
