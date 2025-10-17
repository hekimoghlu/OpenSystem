/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 10, 2023.
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
#include <WebCore/FetchLoader.h>
#include <WebCore/FetchLoaderClient.h>
#include <WebCore/NetworkLoadMetrics.h>
#include <WebCore/ResourceResponse.h>
#include <WebCore/ServiceWorkerFetch.h>
#include <WebCore/ServiceWorkerTypes.h>
#include <WebCore/SharedBuffer.h>
#include <wtf/Lock.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/UniqueRef.h>

namespace WebKit {

class WebServiceWorkerFetchTaskClient final : public WebCore::ServiceWorkerFetch::Client {
public:
    static Ref<WebServiceWorkerFetchTaskClient> create(Ref<IPC::Connection>&& connection, WebCore::ServiceWorkerIdentifier serviceWorkerIdentifier,  WebCore::SWServerConnectionIdentifier serverConnectionIdentifier, WebCore::FetchIdentifier fetchTaskIdentifier, bool needsContinueDidReceiveResponseMessage)
    {
        return adoptRef(*new WebServiceWorkerFetchTaskClient(WTFMove(connection), serviceWorkerIdentifier, serverConnectionIdentifier, fetchTaskIdentifier, needsContinueDidReceiveResponseMessage));
    }

    virtual ~WebServiceWorkerFetchTaskClient();

    void continueDidReceiveResponse();
    void convertFetchToDownload();

private:
    WebServiceWorkerFetchTaskClient(Ref<IPC::Connection>&&, WebCore::ServiceWorkerIdentifier, WebCore::SWServerConnectionIdentifier, WebCore::FetchIdentifier, bool needsContinueDidReceiveResponseMessage);

    void didReceiveResponse(const WebCore::ResourceResponse&) final;
    void didReceiveRedirection(const WebCore::ResourceResponse&) final;
    void didReceiveData(const WebCore::SharedBuffer&) final;
    void didReceiveFormDataAndFinish(Ref<WebCore::FormData>&&) final;
    void didFail(const WebCore::ResourceError&) final;
    void didFinish(const WebCore::NetworkLoadMetrics&) final;
    void didNotHandle() final;
    void doCancel() final;
    void setCancelledCallback(Function<void()>&&) final;
    void usePreload() final;
    void contextIsStopping() final;

    void didReceiveDataInternal(const WebCore::SharedBuffer&) WTF_REQUIRES_LOCK(m_connectionLock);
    void didReceiveFormDataAndFinishInternal(Ref<WebCore::FormData>&&) WTF_REQUIRES_LOCK(m_connectionLock);
    void didFailInternal(const WebCore::ResourceError&) WTF_REQUIRES_LOCK(m_connectionLock);
    void didFinishInternal(const WebCore::NetworkLoadMetrics&) WTF_REQUIRES_LOCK(m_connectionLock);
    void didNotHandleInternal() WTF_REQUIRES_LOCK(m_connectionLock);

    void cleanup() WTF_REQUIRES_LOCK(m_connectionLock);

    void didReceiveBlobChunk(const WebCore::SharedBuffer&);
    void didFinishBlobLoading();

    struct BlobLoader final : WebCore::FetchLoaderClient {
        WTF_MAKE_TZONE_ALLOCATED(BlobLoader);
        WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(BlobLoader);
    public:

        explicit BlobLoader(WebServiceWorkerFetchTaskClient& client) : client(client) { }

        // FetchLoaderClient API
        void didReceiveResponse(const WebCore::ResourceResponse&) final { }
        void didReceiveData(const WebCore::SharedBuffer& data) final { client->didReceiveBlobChunk(data); }
        void didFail(const WebCore::ResourceError& error) final { client->didFail(error); }
        void didSucceed(const WebCore::NetworkLoadMetrics&) final { client->didFinishBlobLoading(); }

        const Ref<WebServiceWorkerFetchTaskClient> client;
        std::unique_ptr<WebCore::FetchLoader> loader;
    };

    Lock m_connectionLock;
    RefPtr<IPC::Connection> m_connection WTF_GUARDED_BY_LOCK(m_connectionLock);
    WebCore::SWServerConnectionIdentifier m_serverConnectionIdentifier;
    WebCore::ServiceWorkerIdentifier m_serviceWorkerIdentifier;
    WebCore::FetchIdentifier m_fetchIdentifier;
    std::unique_ptr<BlobLoader> m_blobLoader;
    bool m_needsContinueDidReceiveResponseMessage { false };
    bool m_waitingForContinueDidReceiveResponseMessage { false };
    std::variant<std::nullptr_t, WebCore::SharedBufferBuilder, Ref<WebCore::FormData>, UniqueRef<WebCore::ResourceError>> m_responseData;
    WebCore::NetworkLoadMetrics m_networkLoadMetrics;
    bool m_didFinish { false };
    bool m_isDownload { false };
    bool m_didSendResponse { false };
    Function<void()> m_cancelledCallback;
};

} // namespace WebKit
