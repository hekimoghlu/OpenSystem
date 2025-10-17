/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 16, 2023.
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

#include "Color.h"
#include "FloatRect.h"
#include "LegacyRenderSVGResource.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class RenderObject;

class LegacyRenderSVGResourceSolidColor final : public LegacyRenderSVGResource {
    WTF_MAKE_TZONE_ALLOCATED(LegacyRenderSVGResourceSolidColor);
public:
    LegacyRenderSVGResourceSolidColor();
    virtual ~LegacyRenderSVGResourceSolidColor();

    void removeAllClientsFromCache() override { }
    void removeAllClientsFromCacheAndMarkForInvalidationIfNeeded(bool, SingleThreadWeakHashSet<RenderObject>*) override { }
    void removeClientFromCache(RenderElement&) override { }
    void removeClientFromCacheAndMarkForInvalidation(RenderElement&, bool = true) override { }

    OptionSet<ApplyResult> applyResource(RenderElement&, const RenderStyle&, GraphicsContext*&, OptionSet<RenderSVGResourceMode>) override;
    void postApplyResource(RenderElement&, GraphicsContext*&, OptionSet<RenderSVGResourceMode>, const Path*, const RenderElement*) override;
    FloatRect resourceBoundingBox(const RenderObject&, RepaintRectCalculation) override { return FloatRect(); }

    RenderSVGResourceType resourceType() const override { return SolidColorResourceType; }

    const Color& color() const { return m_color; }
    void setColor(const Color& color) { m_color = color; }

private:
    Color m_color;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_LEGACY_RENDER_SVG_RESOURCE(LegacyRenderSVGResourceSolidColor, SolidColorResourceType)
