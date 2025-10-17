/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 29, 2022.
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

#include "Connection.h"
#include "DrawingAreaInfo.h"
#include "MessageReceiver.h"
#include "MessageSender.h"
#include <WebCore/FloatRect.h>
#include <WebCore/IntRect.h>
#include <WebCore/IntSize.h>
#include <WebCore/ProcessIdentifier.h>
#include <stdint.h>
#include <wtf/AbstractRefCounted.h>
#include <wtf/Identified.h>
#include <wtf/Noncopyable.h>
#include <wtf/RunLoop.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/TypeCasts.h>
#include <wtf/WeakRef.h>

#if PLATFORM(COCOA)
namespace WTF {
class MachSendRight;
}
#endif

namespace WebCore {
enum class DelegatedScrollingMode : uint8_t;
using FramesPerSecond = unsigned;
using PlatformDisplayID = uint32_t;
}

namespace WebKit {

class LayerTreeContext;
class RemotePageDrawingAreaProxy;
class WebPageProxy;
class WebProcessProxy;

#if USE(COORDINATED_GRAPHICS) || USE(TEXTURE_MAPPER)
struct UpdateInfo;
#endif

class DrawingAreaProxy : public IPC::MessageReceiver, public IPC::MessageSender, public Identified<DrawingAreaIdentifier>, public CanMakeCheckedPtr<DrawingAreaProxy> {
    WTF_MAKE_TZONE_ALLOCATED(DrawingAreaProxy);
    WTF_MAKE_NONCOPYABLE(DrawingAreaProxy);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(DrawingAreaProxy);
public:
    virtual ~DrawingAreaProxy();

    DrawingAreaType type() const { return m_type; }

    virtual bool isRemoteLayerTreeDrawingAreaProxyMac() const { return false; }
    virtual bool isRemoteLayerTreeDrawingAreaProxyIOS() const { return false; }

    void startReceivingMessages(WebProcessProxy&);
    void stopReceivingMessages(WebProcessProxy&);
    virtual std::span<IPC::ReceiverName> messageReceiverNames() const;

    virtual WebCore::DelegatedScrollingMode delegatedScrollingMode() const;

    virtual void deviceScaleFactorDidChange(CompletionHandler<void()>&&) = 0;
    virtual void colorSpaceDidChange() { }
    virtual void windowScreenDidChange(WebCore::PlatformDisplayID) { }
    virtual std::optional<WebCore::FramesPerSecond> displayNominalFramesPerSecond() { return std::nullopt; }

    // FIXME: These should be pure virtual.
    virtual void setBackingStoreIsDiscardable(bool) { }

    const WebCore::IntSize& size() const { return m_size; }
    bool setSize(const WebCore::IntSize&, const WebCore::IntSize& scrollOffset = { });

    virtual void minimumSizeForAutoLayoutDidChange() { }
    virtual void sizeToContentAutoSizeMaximumSizeDidChange() { }
    virtual void windowKindDidChange() { }

    virtual void adjustTransientZoom(double, WebCore::FloatPoint) { }
    virtual void commitTransientZoom(double, WebCore::FloatPoint) { }

    virtual void viewIsBecomingVisible() { }
    virtual void viewIsBecomingInvisible() { }

#if PLATFORM(MAC)
    virtual void didChangeViewExposedRect();
    void viewExposedRectChangedTimerFired();
#endif

    virtual void updateDebugIndicator() { }

    virtual void waitForDidUpdateActivityState(ActivityStateChangeID) { }

    // Hide the content until the currently pending update arrives.
    virtual void hideContentUntilPendingUpdate() { ASSERT_NOT_REACHED(); }

    // Hide the content until any update arrives.
    virtual void hideContentUntilAnyUpdate() { ASSERT_NOT_REACHED(); }

    virtual bool hasVisibleContent() const { return true; }

    virtual void prepareForAppSuspension() { }

#if PLATFORM(COCOA)
    virtual WTF::MachSendRight createFence();
#endif

    virtual void dispatchPresentationCallbacksAfterFlushingLayers(IPC::Connection&, Vector<IPC::AsyncReplyID>&&) { }

    virtual bool shouldCoalesceVisualEditorStateUpdates() const { return false; }
    virtual bool shouldSendWheelEventsToEventDispatcher() const { return false; }

    WebPageProxy* page() const;
    virtual void viewWillStartLiveResize() { };
    virtual void viewWillEndLiveResize() { };

    // IPC::MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) override;

    // IPC::MessageSender
    bool sendMessage(UniqueRef<IPC::Encoder>&&, OptionSet<IPC::SendOption>) final;
    bool sendMessageWithAsyncReply(UniqueRef<IPC::Encoder>&&, AsyncReplyHandler, OptionSet<IPC::SendOption>) final;
    IPC::Connection* messageSenderConnection() const final;
    uint64_t messageSenderDestinationID() const final;

    virtual void addRemotePageDrawingAreaProxy(RemotePageDrawingAreaProxy&) { }
    virtual void removeRemotePageDrawingAreaProxy(RemotePageDrawingAreaProxy&) { }

    virtual void remotePageProcessDidTerminate(WebCore::ProcessIdentifier) { }

protected:
    DrawingAreaProxy(DrawingAreaType, WebPageProxy&, WebProcessProxy&);

    RefPtr<WebPageProxy> protectedWebPageProxy() const;
    Ref<WebProcessProxy> protectedWebProcessProxy() const;

    DrawingAreaType m_type;
    WeakPtr<WebPageProxy> m_webPageProxy;
    Ref<WebProcessProxy> m_webProcessProxy;

    WebCore::IntSize m_size;
    WebCore::IntSize m_scrollOffset;

private:
    virtual void sizeDidChange() = 0;

    // Message handlers.
    // FIXME: These should be pure virtual.
    virtual void enterAcceleratedCompositingMode(uint64_t /* backingStoreStateID */, const LayerTreeContext&) { }
    virtual void updateAcceleratedCompositingMode(uint64_t /* backingStoreStateID */, const LayerTreeContext&) { }
    virtual void didFirstLayerFlush(uint64_t /* backingStoreStateID */, const LayerTreeContext&) { }
#if PLATFORM(MAC)
    RunLoop::Timer m_viewExposedRectChangedTimer;
    std::optional<WebCore::FloatRect> m_lastSentViewExposedRect;
#endif // PLATFORM(MAC)

#if USE(COORDINATED_GRAPHICS) || USE(TEXTURE_MAPPER)
    virtual void update(uint64_t /* backingStoreStateID */, UpdateInfo&&) { }
    virtual void exitAcceleratedCompositingMode(uint64_t /* backingStoreStateID */, UpdateInfo&&) { }
#endif
};

} // namespace WebKit

#define SPECIALIZE_TYPE_TRAITS_DRAWING_AREA_PROXY(ToValueTypeName, ProxyType) \
SPECIALIZE_TYPE_TRAITS_BEGIN(WebKit::ToValueTypeName) \
    static bool isType(const WebKit::DrawingAreaProxy& proxy) { return proxy.type() == WebKit::ProxyType; } \
SPECIALIZE_TYPE_TRAITS_END()

