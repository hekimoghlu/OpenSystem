/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 20, 2023.
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

#if ENABLE(RESOURCE_USAGE)

#include "FloatRect.h"
#include "IntRect.h"
#include "PageOverlay.h"
#include <wtf/Noncopyable.h>
#include <wtf/RefCounted.h>
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

#if PLATFORM(COCOA)
#include "PlatformCALayer.h"
#endif

#if OS(LINUX)
#include "GraphicsLayer.h"
#endif

namespace WebCore {

class FloatRect;
class IntPoint;
class IntRect;

class ResourceUsageOverlay final : public PageOverlayClient, public RefCounted<ResourceUsageOverlay>, public CanMakeWeakPtr<ResourceUsageOverlay> {
    WTF_MAKE_TZONE_ALLOCATED(ResourceUsageOverlay);
    WTF_MAKE_NONCOPYABLE(ResourceUsageOverlay);
public:
    static Ref<ResourceUsageOverlay> create(Page&);
    ~ResourceUsageOverlay();

    PageOverlay& overlay() { return *m_overlay; }

#if PLATFORM(COCOA)
    void platformDraw(CGContextRef);
#endif

    void detachFromPage() { m_page.clear(); }

    static const int normalWidth = 570;
    static const int normalHeight = 180;

private:
    explicit ResourceUsageOverlay(Page&);

    void willMoveToPage(PageOverlay&, Page*) override { }
    void didMoveToPage(PageOverlay&, Page*) override { }
    void drawRect(PageOverlay&, GraphicsContext&, const IntRect&) override { }
    bool mouseEvent(PageOverlay&, const PlatformMouseEvent&) override;
    void didScrollFrame(PageOverlay&, LocalFrame&) override { }

    void initialize();

    void platformInitialize();
    void platformDestroy();

    WeakPtr<Page> m_page;
    RefPtr<PageOverlay> m_overlay;
    bool m_dragging { false };
    IntPoint m_dragPoint;

#if PLATFORM(COCOA)
    RetainPtr<CALayer> m_layer;
    RetainPtr<CALayer> m_containerLayer;
#endif

#if OS(LINUX)
    RefPtr<GraphicsLayer> m_paintLayer;
    std::unique_ptr<GraphicsLayerClient> m_overlayPainter;
#endif
};

} // namespace WebCore

#endif // ENABLE(RESOURCE_USAGE)
