/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 12, 2025.
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
#import "RemoteGraphicsContextGLProxy.h"

#import "RemoteRenderingBackendProxy.h"

#if ENABLE(GPU_PROCESS) && ENABLE(WEBGL)
#import "GPUConnectionToWebProcess.h"
#import "GPUProcessConnection.h"
#import "RemoteGraphicsContextGLMessages.h"
#import "WebProcess.h"
#import <WebCore/CVUtilities.h>
#import <WebCore/GraphicsLayerContentsDisplayDelegate.h>
#import <WebCore/IOSurface.h>
#import <WebCore/PlatformCALayer.h>
#import <WebCore/PlatformCALayerDelegatedContents.h>

namespace WebKit {

namespace {

class DisplayBufferFence final : public WebCore::PlatformCALayerDelegatedContentsFence {
public:
    static Ref<DisplayBufferFence> create(IPC::Semaphore&& finishedFenceSemaphore)
    {
        return adoptRef(*new DisplayBufferFence(WTFMove(finishedFenceSemaphore)));
    }

    bool waitFor(Seconds timeout) final
    {
        Locker locker { m_lock };
        if (m_signaled)
            return true;
        m_signaled = m_semaphore.waitFor(timeout);
        return m_signaled;
    }

    void forceSignal()
    {
        Locker locker { m_lock };
        if (m_signaled)
            return;
        m_signaled = true;
        m_semaphore.signal();
    }

private:
    DisplayBufferFence(IPC::Semaphore&& finishedFenceSemaphore)
        : m_semaphore(WTFMove(finishedFenceSemaphore))
    {
    }

    Lock m_lock;
    bool m_signaled WTF_GUARDED_BY_LOCK(m_lock) { false };
    IPC::Semaphore m_semaphore;
};

class DisplayBufferDisplayDelegate final : public WebCore::GraphicsLayerContentsDisplayDelegate {
public:
    static Ref<DisplayBufferDisplayDelegate> create(bool isOpaque)
    {
        return adoptRef(*new DisplayBufferDisplayDelegate(isOpaque));
    }

    // WebCore::GraphicsLayerContentsDisplayDelegate overrides.
    void prepareToDelegateDisplay(WebCore::PlatformCALayer& layer) final
    {
        layer.setOpaque(m_isOpaque);
    }

    void display(WebCore::PlatformCALayer& layer) final
    {
        if (m_displayBuffer)
            layer.setDelegatedContents({ MachSendRight { m_displayBuffer }, m_finishedFence, std::nullopt });
        else
            layer.clearContents();
    }

    WebCore::GraphicsLayer::CompositingCoordinatesOrientation orientation() const final
    {
        return WebCore::GraphicsLayer::CompositingCoordinatesOrientation::BottomUp;
    }

    void setDisplayBuffer(MachSendRight&& displayBuffer, RefPtr<DisplayBufferFence> finishedFence)
    {
        if (!displayBuffer) {
            m_finishedFence = nullptr;
            m_displayBuffer = { };
            return;
        }
        if (m_displayBuffer && displayBuffer.sendRight() == m_displayBuffer.sendRight())
            return;
        m_finishedFence = WTFMove(finishedFence);
        m_displayBuffer = WTFMove(displayBuffer);
    }

private:
    DisplayBufferDisplayDelegate(bool isOpaque)
        : m_isOpaque(isOpaque)
    {
    }

    MachSendRight m_displayBuffer;
    RefPtr<DisplayBufferFence> m_finishedFence;
    const bool m_isOpaque;
};

class RemoteGraphicsContextGLProxyCocoa final : public RemoteGraphicsContextGLProxy {
    WTF_MAKE_FAST_ALLOCATED;
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RemoteGraphicsContextGLProxyCocoa);
public:
    // RemoteGraphicsContextGLProxy overrides.
    RefPtr<WebCore::GraphicsLayerContentsDisplayDelegate> layerContentsDisplayDelegate() final { return m_layerContentsDisplayDelegate.ptr(); }
    void prepareForDisplay() final;
    void forceContextLost() final;

private:
    explicit RemoteGraphicsContextGLProxyCocoa(const WebCore::GraphicsContextGLAttributes& attributes)
        : RemoteGraphicsContextGLProxy(attributes)
        , m_layerContentsDisplayDelegate(DisplayBufferDisplayDelegate::create(!attributes.alpha))
    {
    }

    Ref<DisplayBufferDisplayDelegate> protectedLayerContentsDisplayDelegate() const { return m_layerContentsDisplayDelegate; }

    void addNewFence(Ref<DisplayBufferFence> newFence);
    static constexpr size_t maxPendingFences = 3;
    size_t m_oldestFenceIndex { 0 };
    std::array<RefPtr<DisplayBufferFence>, maxPendingFences> m_frameCompletionFences;

    Ref<DisplayBufferDisplayDelegate> m_layerContentsDisplayDelegate;
    friend class RemoteGraphicsContextGLProxy;
};

void RemoteGraphicsContextGLProxyCocoa::prepareForDisplay()
{
    if (isContextLost())
        return;
    IPC::Semaphore finishedSignaller;
    auto sendResult = sendSync(Messages::RemoteGraphicsContextGL::PrepareForDisplay(finishedSignaller));
    if (!sendResult.succeeded()) {
        markContextLost();
        return;
    }
    auto [displayBufferSendRight] = sendResult.takeReply();
    if (!displayBufferSendRight)
        return;
    auto finishedFence = DisplayBufferFence::create(WTFMove(finishedSignaller));
    addNewFence(finishedFence);
    protectedLayerContentsDisplayDelegate()->setDisplayBuffer(WTFMove(displayBufferSendRight), WTFMove(finishedFence));
}

void RemoteGraphicsContextGLProxyCocoa::forceContextLost()
{
    for (auto fence : m_frameCompletionFences) {
        if (fence)
            fence->forceSignal();
    }
    RemoteGraphicsContextGLProxy::forceContextLost();
}

void RemoteGraphicsContextGLProxyCocoa::addNewFence(Ref<DisplayBufferFence> newFence)
{
    // Record the pending fences so that they can be force signaled when context is lost.
    size_t oldestFenceIndex = m_oldestFenceIndex++ % maxPendingFences;
    std::exchange(m_frameCompletionFences[oldestFenceIndex], WTFMove(newFence));
    // Due to the fence being IPC::Semaphore, we do not have very good way of waiting the fence in two places, compositor and here.
    // Thus we just record maxPendingFences and trust that compositor does not advance too far with multiple frames.
}

}

Ref<RemoteGraphicsContextGLProxy> RemoteGraphicsContextGLProxy::platformCreate(const WebCore::GraphicsContextGLAttributes& attributes)
{
    return adoptRef(*new RemoteGraphicsContextGLProxyCocoa(attributes));
}

}

#endif
