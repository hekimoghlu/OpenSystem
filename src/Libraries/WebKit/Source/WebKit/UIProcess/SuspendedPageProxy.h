/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 23, 2023.
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
#include "ProcessThrottler.h"
#include "WebBackForwardListItem.h"
#include "WebPageProxyMessageReceiverRegistration.h"
#include "WebProcessProxy.h"
#include <WebCore/FrameIdentifier.h>
#include <WebCore/NavigationIdentifier.h>
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class RegistrableDomain;
}

namespace WebKit {

class BrowsingContextGroup;
class RemotePageProxy;
class WebBackForwardCache;
class WebPageProxy;
class WebProcessPool;
class WebsiteDataStore;

#if HAVE(VISIBILITY_PROPAGATION_VIEW)
using LayerHostingContextID = uint32_t;
#endif

enum class ShouldDelayClosingUntilFirstLayerFlush : bool { No, Yes };

class SuspendedPageProxy final: public IPC::MessageReceiver, public RefCounted<SuspendedPageProxy>, public CanMakeCheckedPtr<SuspendedPageProxy> {
    WTF_MAKE_TZONE_ALLOCATED(SuspendedPageProxy);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SuspendedPageProxy);
public:
    static Ref<SuspendedPageProxy> create(WebPageProxy&, Ref<WebProcessProxy>&&, Ref<WebFrameProxy>&& mainFrame, Ref<BrowsingContextGroup>&&, ShouldDelayClosingUntilFirstLayerFlush);
    ~SuspendedPageProxy();

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    static RefPtr<WebProcessProxy> findReusableSuspendedPageProcess(WebProcessPool&, const WebCore::RegistrableDomain&, WebsiteDataStore&, WebProcessProxy::LockdownMode, const API::PageConfiguration&);

    WebPageProxy* page() const;
    WebCore::PageIdentifier webPageID() const { return m_webPageID; }
    WebProcessProxy& process() const { return m_process.get(); }
    Ref<WebProcessProxy> protectedProcess() const { return process(); }
    WebFrameProxy& mainFrame() { return m_mainFrame.get(); }
    BrowsingContextGroup& browsingContextGroup() { return m_browsingContextGroup.get(); }

    WebBackForwardCache& backForwardCache() const;

    bool pageIsClosedOrClosing() const;

    void waitUntilReadyToUnsuspend(CompletionHandler<void(SuspendedPageProxy*)>&&);
    void unsuspend();

    void pageDidFirstLayerFlush();
    void closeWithoutFlashing();

#if HAVE(VISIBILITY_PROPAGATION_VIEW)
    LayerHostingContextID contextIDForVisibilityPropagationInWebProcess() const { return m_contextIDForVisibilityPropagationInWebProcess; }
#if ENABLE(GPU_PROCESS)
    LayerHostingContextID contextIDForVisibilityPropagationInGPUProcess() const { return m_contextIDForVisibilityPropagationInGPUProcess; }
#endif
#endif

#if !LOG_DISABLED
    String loggingString() const;
#endif

private:
    SuspendedPageProxy(WebPageProxy&, Ref<WebProcessProxy>&&, Ref<WebFrameProxy>&& mainFrame, Ref<BrowsingContextGroup>&&, ShouldDelayClosingUntilFirstLayerFlush);

    enum class SuspensionState : uint8_t { Suspending, FailedToSuspend, Suspended, Resumed };
    void didProcessRequestToSuspend(SuspensionState);
    void suspensionTimedOut();

    void close();
    void didDestroyNavigation(WebCore::NavigationIdentifier);

    // IPC::MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;
    bool didReceiveSyncMessage(IPC::Connection&, IPC::Decoder&, UniqueRef<IPC::Encoder>&) final;

    template<typename T> void sendToAllProcesses(T&&);

    WeakPtr<WebPageProxy> m_page;
    WebCore::PageIdentifier m_webPageID;
    Ref<WebProcessProxy> m_process;
    Ref<WebFrameProxy> m_mainFrame;
    Ref<BrowsingContextGroup> m_browsingContextGroup;
    WebPageProxyMessageReceiverRegistration m_messageReceiverRegistration;
    bool m_isClosed { false };
    ShouldDelayClosingUntilFirstLayerFlush m_shouldDelayClosingUntilFirstLayerFlush { ShouldDelayClosingUntilFirstLayerFlush::No };
    bool m_shouldCloseWhenEnteringAcceleratedCompositingMode { false };

    SuspensionState m_suspensionState { SuspensionState::Suspending };
    CompletionHandler<void(SuspendedPageProxy*)> m_readyToUnsuspendHandler;
    RunLoop::Timer m_suspensionTimeoutTimer;
#if USE(RUNNINGBOARD)
    RefPtr<ProcessThrottler::BackgroundActivity> m_suspensionActivity;
#endif
#if HAVE(VISIBILITY_PROPAGATION_VIEW)
    LayerHostingContextID m_contextIDForVisibilityPropagationInWebProcess { 0 };
#if ENABLE(GPU_PROCESS)
    LayerHostingContextID m_contextIDForVisibilityPropagationInGPUProcess { 0 };
#endif
#endif
};

} // namespace WebKit
