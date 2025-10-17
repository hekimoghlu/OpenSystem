/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 20, 2022.
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

#if USE(COORDINATED_GRAPHICS)
#include "CompositingRunLoop.h"
#include <WebCore/Damage.h>
#include <WebCore/DisplayUpdate.h>
#include <WebCore/GLContext.h>
#include <WebCore/IntSize.h>
#include <WebCore/TextureMapperFPSCounter.h>
#include <atomic>
#include <wtf/Atomics.h>
#include <wtf/CheckedPtr.h>
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeRefCounted.h>

#if !HAVE(DISPLAY_LINK)
#include "ThreadedDisplayRefreshMonitor.h"
#endif

namespace WebCore {
class TextureMapper;
class TransformationMatrix;
}

namespace WebKit {
class AcceleratedSurface;
class CoordinatedSceneState;
class LayerTreeHost;

class ThreadedCompositor : public ThreadSafeRefCounted<ThreadedCompositor>, public CanMakeThreadSafeCheckedPtr<ThreadedCompositor> {
    WTF_MAKE_TZONE_ALLOCATED(ThreadedCompositor);
    WTF_MAKE_NONCOPYABLE(ThreadedCompositor);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(ThreadedCompositor);
public:
    enum class DamagePropagation : uint8_t {
        None,
        Region,
        Unified,
    };

#if HAVE(DISPLAY_LINK)
    static Ref<ThreadedCompositor> create(LayerTreeHost&);
#else
    static Ref<ThreadedCompositor> create(LayerTreeHost&, ThreadedDisplayRefreshMonitor::Client&, WebCore::PlatformDisplayID);
#endif
    virtual ~ThreadedCompositor();

    uint64_t surfaceID() const;

    void backgroundColorDidChange();
#if PLATFORM(WPE) && USE(GBM) && ENABLE(WPE_PLATFORM)
    void preferredBufferFormatsDidChange();
#endif

    uint32_t requestComposition();
    void scheduleUpdate();
    RunLoop* runLoop();

    void invalidate();

#if !HAVE(DISPLAY_LINK)
    WebCore::DisplayRefreshMonitor& displayRefreshMonitor() const;
#endif

    void suspend();
    void resume();

    bool isActive() const;

#if ENABLE(DAMAGE_TRACKING)
    void setDamagePropagation(WebCore::Damage::Propagation);
#endif

private:
#if HAVE(DISPLAY_LINK)
    explicit ThreadedCompositor(LayerTreeHost&);
#else
    ThreadedCompositor(LayerTreeHost&, ThreadedDisplayRefreshMonitor::Client&, WebCore::PlatformDisplayID);
#endif

    void updateSceneState();
    void renderLayerTree();
    void paintToCurrentGLContext(const WebCore::TransformationMatrix&, const WebCore::IntSize&);
    void frameComplete();

#if HAVE(DISPLAY_LINK)
    void didRenderFrameTimerFired();
#else
    void displayUpdateFired();
    void sceneUpdateFinished();
#endif

    CheckedPtr<LayerTreeHost> m_layerTreeHost;
    std::unique_ptr<AcceleratedSurface> m_surface;
    RefPtr<CoordinatedSceneState> m_sceneState;
    std::unique_ptr<WebCore::GLContext> m_context;

    bool m_flipY { false };
    std::atomic<unsigned> m_suspendedCount { 0 };

    std::unique_ptr<CompositingRunLoop> m_compositingRunLoop;

    struct {
        Lock lock;
        WebCore::IntSize viewportSize;
        float deviceScaleFactor { 1 };
        uint32_t compositionRequestID { 0 };

#if !HAVE(DISPLAY_LINK)
        bool clientRendersNextFrame { false };
#endif
    } m_attributes;

    std::unique_ptr<WebCore::TextureMapper> m_textureMapper;
    WebCore::TextureMapperFPSCounter m_fpsCounter;

#if ENABLE(DAMAGE_TRACKING)
    WebCore::Damage::Propagation m_damagePropagation { WebCore::Damage::Propagation::None };
#endif

#if HAVE(DISPLAY_LINK)
    std::atomic<uint32_t> m_compositionResponseID { 0 };
    RunLoop::Timer m_didRenderFrameTimer;
#else
    struct {
        WebCore::PlatformDisplayID displayID;
        WebCore::DisplayUpdate displayUpdate;
        std::unique_ptr<RunLoop::Timer> updateTimer;
    } m_display;

    Ref<ThreadedDisplayRefreshMonitor> m_displayRefreshMonitor;
#endif
};

} // namespace WebKit

#endif // USE(COORDINATED_GRAPHICS)

