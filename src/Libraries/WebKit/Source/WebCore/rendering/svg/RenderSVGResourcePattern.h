/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 10, 2022.
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
#include "PatternAttributes.h"
#include "RenderSVGResourcePaintServer.h"
#include "SVGPatternElement.h"

class Pattern;

namespace WebCore {

class RenderSVGResourcePattern final : public RenderSVGResourcePaintServer {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderSVGResourcePattern);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderSVGResourcePattern);
public:
    RenderSVGResourcePattern(SVGElement&, RenderStyle&&);
    virtual ~RenderSVGResourcePattern();

    inline SVGPatternElement& patternElement() const;
    inline Ref<SVGPatternElement> protectedPatternElement() const;

    bool prepareFillOperation(GraphicsContext&, const RenderLayerModelObject&, const RenderStyle&) final;
    bool prepareStrokeOperation(GraphicsContext&, const RenderLayerModelObject&, const RenderStyle&) final;

    enum class SuppressRepaint { Yes, No };
    void invalidatePattern(SuppressRepaint suppressRepaint = SuppressRepaint::No)
    {
        m_attributes = std::nullopt;
        m_imageMap.clear();
        m_transformMap.clear();
        if (suppressRepaint == SuppressRepaint::No)
            repaintAllClients();
    }

protected:
    RefPtr<Pattern> buildPattern(GraphicsContext&, const RenderLayerModelObject&);

    void collectPatternAttributesIfNeeded();

    bool buildTileImageTransform(const RenderElement&, const PatternAttributes&, const SVGPatternElement&, FloatRect& patternBoundaries, AffineTransform& tileImageTransform) const;

    RefPtr<ImageBuffer> createTileImage(GraphicsContext&, const PatternAttributes&, const FloatSize&, const FloatSize& scale, const AffineTransform& tileImageTransform) const;

    void removeReferencingCSSClient(const RenderElement&) override;

    std::optional<PatternAttributes> m_attributes;

    UncheckedKeyHashMap<SingleThreadWeakRef<const RenderLayerModelObject>, RefPtr<ImageBuffer>> m_imageMap;
    UncheckedKeyHashMap<SingleThreadWeakRef<const RenderLayerModelObject>, AffineTransform> m_transformMap;
};

}

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderSVGResourcePattern, isRenderSVGResourcePattern())
