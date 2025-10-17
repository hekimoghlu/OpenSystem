/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 29, 2023.
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
#include <memory>
#include <wtf/HashMap.h>
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakRef.h>

namespace WebCore {

class LegacyRenderSVGResourceContainer;
class RenderElement;
class RenderObject;
class RenderStyle;
class SVGResources;

class SVGResourcesCache {
    WTF_MAKE_TZONE_ALLOCATED(SVGResourcesCache);
    WTF_MAKE_NONCOPYABLE(SVGResourcesCache);
public:
    SVGResourcesCache();
    ~SVGResourcesCache();

    static SVGResources* cachedResourcesForRenderer(const RenderElement&);

    // Called from all SVG renderers addChild() methods.
    static void clientWasAddedToTree(RenderObject&);

    // Called from all SVG renderers removeChild() methods.
    static void clientWillBeRemovedFromTree(RenderObject&);

    // Called from all SVG renderers destroy() methods - except for LegacyRenderSVGResourceContainer.
    static void clientDestroyed(RenderElement&);

    // Called from all SVG renderers layout() methods.
    static void clientLayoutChanged(RenderElement&);

    // Called from all SVG renderers styleDidChange() methods.
    static void clientStyleChanged(RenderElement&, StyleDifference, const RenderStyle* oldStyle, const RenderStyle& newStyle);

    // Called from LegacyRenderSVGResourceContainer::willBeDestroyed().
    static void resourceDestroyed(LegacyRenderSVGResourceContainer&);

    class SetStyleForScope {
        WTF_MAKE_NONCOPYABLE(SetStyleForScope);
    public:
        SetStyleForScope(RenderElement&, const RenderStyle& scopedStyle, const RenderStyle& newStyle);
        ~SetStyleForScope();
    private:
        void setStyle(const RenderStyle&);

        RenderElement& m_renderer;
        const RenderStyle& m_scopedStyle;
        bool m_needsNewStyle { false };
    };

private:
    void addResourcesFromRenderer(RenderElement&, const RenderStyle&);
    void removeResourcesFromRenderer(RenderElement&);

    using CacheMap = UncheckedKeyHashMap<SingleThreadWeakRef<const RenderElement>, std::unique_ptr<SVGResources>>;
    CacheMap m_cache;
};

} // namespace WebCore
