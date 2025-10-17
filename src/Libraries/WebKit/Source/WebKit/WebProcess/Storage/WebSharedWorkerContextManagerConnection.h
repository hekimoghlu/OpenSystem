/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 24, 2022.
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
#include "IdentifierTypes.h"
#include "MessageReceiver.h"
#include "WebPageProxyIdentifier.h"
#include "WebPreferencesStore.h"
#include <WebCore/PageIdentifier.h>
#include <WebCore/SharedWorkerContextManager.h>
#include <WebCore/SharedWorkerIdentifier.h>
#include <WebCore/Site.h>
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {
struct ClientOrigin;
struct WorkerFetchResult;
struct WorkerInitializationData;
struct WorkerOptions;
}

namespace WebKit {

class WebUserContentController;
struct RemoteWorkerInitializationData;

class WebSharedWorkerContextManagerConnection final : public WebCore::SharedWorkerContextManager::Connection, public IPC::MessageReceiver, public RefCounted<WebSharedWorkerContextManagerConnection> {
    WTF_MAKE_TZONE_ALLOCATED(WebSharedWorkerContextManagerConnection);
public:
    static Ref<WebSharedWorkerContextManagerConnection> create(Ref<IPC::Connection>&&, WebCore::Site&&, PageGroupIdentifier, WebPageProxyIdentifier, WebCore::PageIdentifier, const WebPreferencesStore&, RemoteWorkerInitializationData&&);
    ~WebSharedWorkerContextManagerConnection();

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    void establishConnection(CompletionHandler<void()>&&) final;
    void postErrorToWorkerObject(WebCore::SharedWorkerIdentifier, const String& errorMessage, int lineNumber, int columnNumber, const String& sourceURL, bool isErrorEvent) final;
    void sharedWorkerTerminated(WebCore::SharedWorkerIdentifier) final;

    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;

private:
    WebSharedWorkerContextManagerConnection(Ref<IPC::Connection>&&, WebCore::Site&&, PageGroupIdentifier, WebPageProxyIdentifier, WebCore::PageIdentifier, const WebPreferencesStore&, RemoteWorkerInitializationData&&);

    // IPC Messages.
    void launchSharedWorker(WebCore::ClientOrigin&&, WebCore::SharedWorkerIdentifier, WebCore::WorkerOptions&&, WebCore::WorkerFetchResult&&, WebCore::WorkerInitializationData&&);
    void updatePreferencesStore(const WebPreferencesStore&);
    void setUserAgent(String&& userAgent) { m_userAgent = WTFMove(userAgent); }
    void close();

    Ref<IPC::Connection> m_connectionToNetworkProcess;
    const WebCore::Site m_site;
    PageGroupIdentifier m_pageGroupID;
    WebPageProxyIdentifier m_webPageProxyID;
    WebCore::PageIdentifier m_pageID;
    String m_userAgent;
    Ref<WebUserContentController> m_userContentController;
    std::optional<WebPreferencesStore> m_preferencesStore;
};

} // namespace WebKit
