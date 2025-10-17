/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 14, 2023.
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
#include "LegacyRenderSVGResourceContainer.h"
#include "SVGUnitTypes.h"

#include <wtf/EnumeratedArray.h>
#include <wtf/HashMap.h>

namespace WebCore {

class GraphicsContext;
class SVGMaskElement;

struct MaskerData {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;
    RefPtr<ImageBuffer> maskImage;
};

class LegacyRenderSVGResourceMasker final : public LegacyRenderSVGResourceContainer {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(LegacyRenderSVGResourceMasker);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(LegacyRenderSVGResourceMasker);
public:
    LegacyRenderSVGResourceMasker(SVGMaskElement&, RenderStyle&&);
    virtual ~LegacyRenderSVGResourceMasker();

    inline SVGMaskElement& maskElement() const;
    inline Ref<SVGMaskElement> protectedMaskElement() const;

    void removeAllClientsFromCache() override;
    void removeClientFromCache(RenderElement&) override;
    OptionSet<ApplyResult> applyResource(RenderElement&, const RenderStyle&, GraphicsContext*&, OptionSet<RenderSVGResourceMode>) override;
    bool drawContentIntoContext(GraphicsContext&, const FloatRect& objectBoundingBox);
    bool drawContentIntoContext(GraphicsContext&, const FloatRect& destinationRect, const FloatRect& sourceRect, ImagePaintingOptions);
    FloatRect resourceBoundingBox(const RenderObject&, RepaintRectCalculation) override;

    inline SVGUnitTypes::SVGUnitType maskUnits() const;
    inline SVGUnitTypes::SVGUnitType maskContentUnits() const;

    RenderSVGResourceType resourceType() const override { return MaskerResourceType; }

private:
    void element() const = delete;

    ASCIILiteral renderName() const override { return "RenderSVGResourceMasker"_s; }

    bool drawContentIntoMaskImage(MaskerData*, const DestinationColorSpace&, RenderObject*);
    void calculateMaskContentRepaintRect(RepaintRectCalculation);

    EnumeratedArray<RepaintRectCalculation, FloatRect, RepaintRectCalculation::Accurate> m_maskContentBoundaries;
    UncheckedKeyHashMap<SingleThreadWeakRef<RenderObject>, std::unique_ptr<MaskerData>> m_masker;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_LEGACY_RENDER_SVG_RESOURCE(LegacyRenderSVGResourceMasker, MaskerResourceType)
