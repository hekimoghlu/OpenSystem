/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 12, 2024.
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
#include "MessageReceiver.h"
#include <WebCore/FrameIdentifier.h>
#include <WebCore/InspectorClient.h>
#include <wtf/Noncopyable.h>
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/text/WTFString.h>

namespace WebKit {

class WebPage;

class WebInspector : public ThreadSafeRefCounted<WebInspector>, private IPC::Connection::Client {
    WTF_MAKE_FAST_ALLOCATED;
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(WebInspector);
public:
    static Ref<WebInspector> create(WebPage&);
    ~WebInspector();

    void ref() const final { ThreadSafeRefCounted::ref(); }
    void deref() const final { ThreadSafeRefCounted::deref(); }

    WebPage* page() const;

    void updateDockingAvailability();

    // Implemented in generated WebInspectorMessageReceiver.cpp
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) override;

    // IPC::Connection::Client
    void didClose(IPC::Connection&) override { close(); }
    void didReceiveInvalidMessage(IPC::Connection&, IPC::MessageName, int32_t indexOfObjectFailingDecoding) override { close(); }

    void show(CompletionHandler<void()>&&);
    void close();

    void canAttachWindow(bool& result);

    void showConsole();
    void showResources();

    void showMainResourceForFrame(WebCore::FrameIdentifier);

    void setAttached(bool attached) { m_attached = attached; }

    void evaluateScriptForTest(const String& script);

    void startPageProfiling();
    void stopPageProfiling();

    void startElementSelection();
    void stopElementSelection();
    void elementSelectionChanged(bool);
    void timelineRecordingChanged(bool);

    void setDeveloperPreferenceOverride(WebCore::InspectorClient::DeveloperPreference, std::optional<bool>);
#if ENABLE(INSPECTOR_NETWORK_THROTTLING)
    void setEmulatedConditions(std::optional<int64_t>&& bytesPerSecondLimit);
#endif

    void setFrontendConnection(IPC::Connection::Handle&&);

    void disconnectFromPage() { close(); }

private:
    friend class WebInspectorClient;

    explicit WebInspector(WebPage&);

    bool canAttachWindow();

    // Called from WebInspectorClient
    void openLocalInspectorFrontend();
    void closeFrontendConnection();

    void bringToFront();

    void whenFrontendConnectionEstablished(Function<void()>&&);

    WeakPtr<WebPage> m_page;

    RefPtr<IPC::Connection> m_frontendConnection;
    Vector<Function<void()>> m_frontendConnectionActions;

    bool m_attached { false };
    bool m_previousCanAttach { false };
};

} // namespace WebKit
