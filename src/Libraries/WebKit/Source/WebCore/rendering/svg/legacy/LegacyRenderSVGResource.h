/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 27, 2022.
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

#include "RenderStyleConstants.h"
#include <wtf/OptionSet.h>
#include <wtf/TypeCasts.h>
#include <wtf/WeakHashSet.h>

namespace WebCore {

enum RenderSVGResourceType {
    MaskerResourceType,
    MarkerResourceType,
    PatternResourceType,
    LinearGradientResourceType,
    RadialGradientResourceType,
    SolidColorResourceType,
    FilterResourceType,
    ClipperResourceType
};

// If this enum changes change the unsigned bitfields using it.
enum class RenderSVGResourceMode {
    ApplyToFill    = 1 << 0,
    ApplyToStroke  = 1 << 1,
    ApplyToText    = 1 << 2 // used in combination with ApplyTo{Fill|Stroke}Mode
};

enum class RepaintRectCalculation : bool;

class Color;
class FloatRect;
class GraphicsContext;
class LegacyRenderSVGResourceSolidColor;
class Path;
class RenderElement;
class RenderObject;
class RenderStyle;

class LegacyRenderSVGResource {
public:
    LegacyRenderSVGResource() = default;
    virtual ~LegacyRenderSVGResource() = default;

    void removeAllClientsFromCacheAndMarkForInvalidation(bool markForInvalidation = true);
    virtual void removeAllClientsFromCache() = 0;
    virtual void removeAllClientsFromCacheAndMarkForInvalidationIfNeeded(bool markForInvalidation, SingleThreadWeakHashSet<RenderObject>* visitedRenderers) = 0;
    virtual void removeClientFromCache(RenderElement&) = 0;
    virtual void removeClientFromCacheAndMarkForInvalidation(RenderElement&, bool markForInvalidation = true) = 0;

    enum class ApplyResult : uint8_t {
        ResourceApplied = 1 << 0,
        ClipContainsRendererContent = 1 << 1
    };

    virtual OptionSet<ApplyResult> applyResource(RenderElement&, const RenderStyle&, GraphicsContext*&, OptionSet<RenderSVGResourceMode>) = 0;
    virtual void postApplyResource(RenderElement&, GraphicsContext*&, OptionSet<RenderSVGResourceMode>, const Path*, const RenderElement* /* shape */) { }
    virtual FloatRect resourceBoundingBox(const RenderObject&, RepaintRectCalculation) = 0;

    virtual RenderSVGResourceType resourceType() const = 0;

    // Helper utilities used in the render tree to access resources used for painting shapes/text (gradients & patterns & solid colors only)
    static LegacyRenderSVGResource* fillPaintingResource(RenderElement&, const RenderStyle&, Color& fallbackColor);
    static LegacyRenderSVGResource* strokePaintingResource(RenderElement&, const RenderStyle&, Color& fallbackColor);
    static LegacyRenderSVGResourceSolidColor* sharedSolidPaintingResource();

    static void markForLayoutAndParentResourceInvalidation(RenderObject&, bool needsLayout = true);
    static void markForLayoutAndParentResourceInvalidationIfNeeded(RenderObject&, bool needsLayout, SingleThreadWeakHashSet<RenderObject>* visitedRenderers);

    static void fillAndStrokePathOrShape(GraphicsContext&, OptionSet<RenderSVGResourceMode>, const Path*, const RenderElement* shape);
};

constexpr bool resourceWasApplied(OptionSet<LegacyRenderSVGResource::ApplyResult> result)
{
    return result.contains(LegacyRenderSVGResource::ApplyResult::ResourceApplied);
}

} // namespace WebCore

#define SPECIALIZE_TYPE_TRAITS_LEGACY_RENDER_SVG_RESOURCE(ToValueTypeName, ResourceType) \
SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::ToValueTypeName) \
    static bool isType(const WebCore::LegacyRenderSVGResource& resource) { return resource.resourceType() == WebCore::ResourceType; } \
SPECIALIZE_TYPE_TRAITS_END()
