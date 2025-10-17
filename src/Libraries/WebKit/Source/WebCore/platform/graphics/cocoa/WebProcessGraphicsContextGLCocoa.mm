/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 30, 2024.
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
#import "config.h"

#if ENABLE(WEBGL)
#import "GraphicsContextGLCocoa.h" // NOLINT
#import "GraphicsLayerContentsDisplayDelegate.h"
#import "PlatformCALayer.h"
#import "PlatformCALayerDelegatedContents.h"
#import "ProcessIdentity.h"
#import <wtf/Condition.h>
#import <wtf/Lock.h>
#import <wtf/Noncopyable.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

constexpr Seconds frameFinishedTimeout = 5_s;

namespace {

class DisplayBufferFence final : public PlatformCALayerDelegatedContentsFence {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(DisplayBufferFence);
    WTF_MAKE_NONCOPYABLE(DisplayBufferFence);
public:
    static RefPtr<DisplayBufferFence> create()
    {
        return adoptRef(new DisplayBufferFence);
    }

    bool waitFor(Seconds timeout) final
    {
        Locker locker { m_lock };
        auto absoluteTime = MonotonicTime::timePointFromNow(timeout);
        return m_condition.waitUntil(m_lock, absoluteTime, [&] {
            assertIsHeld(m_lock);
            return m_isSet;
        });
    }

    void signalAll()
    {
        Locker locker { m_lock };
        if (m_isSet)
            return;
        m_isSet = true;
        m_condition.notifyAll();
    }

private:
    DisplayBufferFence() = default;
    Lock m_lock;
    bool m_isSet WTF_GUARDED_BY_LOCK(m_lock) { false };
    Condition m_condition;
};

class DisplayBufferDisplayDelegate final : public GraphicsLayerContentsDisplayDelegate {
public:
    static Ref<DisplayBufferDisplayDelegate> create(bool isOpaque)
    {
        return adoptRef(*new DisplayBufferDisplayDelegate(isOpaque));
    }

    // GraphicsLayerContentsDisplayDelegate overrides.
    void prepareToDelegateDisplay(PlatformCALayer& layer) final
    {
        layer.setOpaque(m_isOpaque);
    }

    void display(PlatformCALayer& layer) final
    {
        if (m_displayBuffer)
            layer.setDelegatedContents({ *m_displayBuffer, m_finishedFence });
        else
            layer.clearContents();
    }

    GraphicsLayer::CompositingCoordinatesOrientation orientation() const final
    {
        return GraphicsLayer::CompositingCoordinatesOrientation::BottomUp;
    }

    void setDisplayBuffer(IOSurface* displayBuffer, RefPtr<DisplayBufferFence> finishedFence)
    {
        if (!displayBuffer) {
            m_finishedFence = nullptr;
            m_displayBuffer.reset();
            return;
        }
        if (m_displayBuffer && displayBuffer->surface() == m_displayBuffer->surface())
            return;
        m_displayBuffer = IOSurface::createFromSurface(displayBuffer->surface(), { });
        m_finishedFence = WTFMove(finishedFence);
    }

private:
    DisplayBufferDisplayDelegate(bool isOpaque)
        : m_isOpaque(isOpaque)
    {
    }

    std::unique_ptr<IOSurface> m_displayBuffer;
    RefPtr<DisplayBufferFence> m_finishedFence;
    const bool m_isOpaque;
};

// GraphicsContextGL type that is used when WebGL is run in-process in WebContent process.
class WebProcessGraphicsContextGLCocoa final : public GraphicsContextGLCocoa
{
public:
    ~WebProcessGraphicsContextGLCocoa();

    // GraphicsContextGLCocoa overrides.
    RefPtr<GraphicsLayerContentsDisplayDelegate> layerContentsDisplayDelegate() final { return m_layerContentsDisplayDelegate.ptr(); }
    void prepareForDisplay() final;
private:
    WebProcessGraphicsContextGLCocoa(GraphicsContextGLAttributes&&);
    Ref<DisplayBufferDisplayDelegate> m_layerContentsDisplayDelegate;

    friend RefPtr<GraphicsContextGL> WebCore::createWebProcessGraphicsContextGL(const GraphicsContextGLAttributes&);
    friend class GraphicsContextGLOpenGL;
};

WebProcessGraphicsContextGLCocoa::WebProcessGraphicsContextGLCocoa(GraphicsContextGLAttributes&& attributes)
    : GraphicsContextGLCocoa(WTFMove(attributes), { })
    , m_layerContentsDisplayDelegate(DisplayBufferDisplayDelegate::create(!attributes.alpha))
{
}

WebProcessGraphicsContextGLCocoa::~WebProcessGraphicsContextGLCocoa() = default;

void WebProcessGraphicsContextGLCocoa::prepareForDisplay()
{
    auto finishedFence = DisplayBufferFence::create();
    prepareForDisplayWithFinishedSignal([finishedFence] {
        finishedFence->signalAll();
    });
    // Here we do not record the finishedFence to be force signalled when context is lost.
    // Currently there's no mechanism to detect if scheduled commands were lost, so we
    // assume that scheduled fence will always be signalled. 
    // Here we trust that compositor does not advance too far with multiple frames.
    m_layerContentsDisplayDelegate->setDisplayBuffer(displayBufferSurface(), WTFMove(finishedFence));
}

}

RefPtr<GraphicsContextGL> createWebProcessGraphicsContextGL(const GraphicsContextGLAttributes& attributes)
{
    auto context = adoptRef(new WebProcessGraphicsContextGLCocoa(GraphicsContextGLAttributes { attributes }));
    if (!context->initialize())
        return nullptr;
    return context;
}

}

#endif
