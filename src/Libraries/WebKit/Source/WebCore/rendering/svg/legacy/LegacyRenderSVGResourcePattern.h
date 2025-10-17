/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 30, 2022.
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
#include "Pattern.h"
#include "PatternAttributes.h"
#include "SVGPatternElement.h"
#include <memory>
#include <wtf/HashMap.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

struct PatternData {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(PatternData);
public:
    RefPtr<Pattern> pattern;
    AffineTransform transform;
};

class LegacyRenderSVGResourcePattern final : public LegacyRenderSVGResourceContainer {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(LegacyRenderSVGResourcePattern);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(LegacyRenderSVGResourcePattern);
public:
    LegacyRenderSVGResourcePattern(SVGPatternElement&, RenderStyle&&);
    virtual ~LegacyRenderSVGResourcePattern();

    SVGPatternElement& patternElement() const;
    Ref<SVGPatternElement> protectedPatternElement() const;

    void removeAllClientsFromCache() override;
    void removeAllClientsFromCacheAndMarkForInvalidationIfNeeded(bool markForInvalidation, SingleThreadWeakHashSet<RenderObject>* visitedRenderers) override;
    void removeClientFromCache(RenderElement&) override;

    OptionSet<ApplyResult> applyResource(RenderElement&, const RenderStyle&, GraphicsContext*&, OptionSet<RenderSVGResourceMode>) override;
    void postApplyResource(RenderElement&, GraphicsContext*&, OptionSet<RenderSVGResourceMode>, const Path*, const RenderElement*) override;
    FloatRect resourceBoundingBox(const RenderObject&, RepaintRectCalculation) override { return FloatRect(); }

    RenderSVGResourceType resourceType() const override { return PatternResourceType; }

    void collectPatternAttributes(PatternAttributes&) const;

private:
    void element() const = delete;
    ASCIILiteral renderName() const override { return "RenderSVGResourcePattern"_s; }

    bool buildTileImageTransform(RenderElement&, const PatternAttributes&, const SVGPatternElement&, FloatRect& patternBoundaries, AffineTransform& tileImageTransform) const;

    RefPtr<ImageBuffer> createTileImage(GraphicsContext&, const FloatSize&, const FloatSize& scale, const AffineTransform& tileImageTransform, const PatternAttributes&) const;

    PatternData* buildPattern(RenderElement&, OptionSet<RenderSVGResourceMode>, GraphicsContext&);

    PatternAttributes m_attributes;
    UncheckedKeyHashMap<SingleThreadWeakRef<RenderElement>, std::unique_ptr<PatternData>> m_patternMap;
    bool m_shouldCollectPatternAttributes { true };
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_LEGACY_RENDER_SVG_RESOURCE(LegacyRenderSVGResourcePattern, PatternResourceType)
