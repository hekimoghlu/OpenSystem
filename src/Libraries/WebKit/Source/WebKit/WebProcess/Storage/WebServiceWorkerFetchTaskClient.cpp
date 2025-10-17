/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 26, 2025.
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
#include "config.h"
#include "WebServiceWorkerFetchTaskClient.h"

#include "FormDataReference.h"
#include "Logging.h"
#include "ServiceWorkerDownloadTaskMessages.h"
#include "ServiceWorkerFetchTaskMessages.h"
#include "SharedBufferReference.h"
#include "WebErrors.h"
#include <WebCore/FetchEvent.h>
#include <WebCore/ResourceError.h>
#include <WebCore/ResourceResponse.h>
#include <WebCore/SWContextManager.h>
#include <wtf/RunLoop.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebServiceWorkerFetchTaskClient::BlobLoader);

WebServiceWorkerFetchTaskClient::WebServiceWorkerFetchTaskClient(Ref<IPC::Connection>&& connection, WebCore::ServiceWorkerIdentifier serviceWorkerIdentifier, WebCore::SWServerConnectionIdentifier serverConnectionIdentifier, FetchIdentifier fetchIdentifier, bool needsContinueDidReceiveResponseMessage)
    : m_connection(WTFMove(connection))
    , m_serverConnectionIdentifier(serverConnectionIdentifier)
    , m_serviceWorkerIdentifier(serviceWorkerIdentifier)
    , m_fetchIdentifier(fetchIdentifier)
    , m_needsContinueDidReceiveResponseMessage(needsContinueDidReceiveResponseMessage)
{
}

WebServiceWorkerFetchTaskClient::~WebServiceWorkerFetchTaskClient() = default;

void WebServiceWorkerFetchTaskClient::didReceiveRedirection(const WebCore::ResourceResponse& response)
{
    Locker lock(m_connectionLock);

    RefPtr connection = m_connection;
    if (!connection)
        return;

    m_didSendResponse = true;
    connection->send(Messages::ServiceWorkerFetchTask::DidReceiveRedirectResponse { response }, m_fetchIdentifier);

    cleanup();
}

void WebServiceWorkerFetchTaskClient::didReceiveResponse(const ResourceResponse& response)
{
    Locker lock(m_connectionLock);

    RefPtr connection = m_connection;
    if (!connection)
        return;

    m_didSendResponse = true;
    if (m_needsContinueDidReceiveResponseMessage)
        m_waitingForContinueDidReceiveResponseMessage = true;

    connection->send(Messages::ServiceWorkerFetchTask::DidReceiveResponse { response, m_needsContinueDidReceiveResponseMessage }, m_fetchIdentifier);
}

void WebServiceWorkerFetchTaskClient::didReceiveData(const SharedBuffer& buffer)
{
    Locker lock(m_connectionLock);
    didReceiveDataInternal(buffer);
}

void WebServiceWorkerFetchTaskClient::didReceiveDataInternal(const SharedBuffer& buffer)
{
    RefPtr connection = m_connection;
    if (!connection)
        return;

    if (m_waitingForContinueDidReceiveResponseMessage) {
        if (!std::holds_alternative<SharedBufferBuilder>(m_responseData))
            m_responseData = SharedBufferBuilder();
        std::get<SharedBufferBuilder>(m_responseData).append(buffer);
        return;
    }

    if (m_isDownload)
        connection->send(Messages::ServiceWorkerDownloadTask::DidReceiveData { IPC::SharedBufferReference(buffer), buffer.size() }, m_fetchIdentifier);
    else
        connection->send(Messages::ServiceWorkerFetchTask::DidReceiveData { IPC::SharedBufferReference(buffer), buffer.size() }, m_fetchIdentifier);
}

void WebServiceWorkerFetchTaskClient::didReceiveFormDataAndFinish(Ref<FormData>&& formData)
{
    Locker lock(m_connectionLock);
    didReceiveFormDataAndFinishInternal(WTFMove(formData));
}

void WebServiceWorkerFetchTaskClient::didReceiveFormDataAndFinishInternal(Ref<FormData>&& formData)
{
    if (auto sharedBuffer = formData->asSharedBuffer()) {
        didReceiveDataInternal(sharedBuffer.releaseNonNull());
        didFinishInternal({ });
        return;
    }

    RefPtr connection = m_connection;
    if (!connection)
        return;

    if (m_waitingForContinueDidReceiveResponseMessage) {
        m_responseData = formData->isolatedCopy();
        return;
    }

    // FIXME: We should send this form data to the other process and consume it there.
    // For now and for the case of blobs, we read it there and send the data through IPC.
    URL blobURL = formData->asBlobURL();
    if (blobURL.isNull()) {
        if (m_isDownload)
            connection->send(Messages::ServiceWorkerDownloadTask::DidReceiveFormData { IPC::FormDataReference { WTFMove(formData) } }, m_fetchIdentifier);
        else
            connection->send(Messages::ServiceWorkerFetchTask::DidReceiveFormData { IPC::FormDataReference { WTFMove(formData) } }, m_fetchIdentifier);
        return;
    }

    callOnMainRunLoop([this, protectedThis = Ref { *this }, blobURL = WTFMove(blobURL).isolatedCopy()] () {
        RefPtr serviceWorkerThreadProxy = SWContextManager::singleton().serviceWorkerThreadProxy(m_serviceWorkerIdentifier);
        if (!serviceWorkerThreadProxy) {
            didFail(internalError(blobURL));
            return;
        }

        m_blobLoader = makeUnique<BlobLoader>(*this);
        auto loader = serviceWorkerThreadProxy->createBlobLoader(*m_blobLoader, blobURL);
        if (!loader) {
            m_blobLoader = nullptr;
            didFail(internalError(blobURL));
            return;
        }

        m_blobLoader->loader = WTFMove(loader);
    });
}

void WebServiceWorkerFetchTaskClient::didReceiveBlobChunk(const SharedBuffer& buffer)
{
    Locker lock(m_connectionLock);

    RefPtr connection = m_connection;
    if (!connection)
        return;

    if (m_isDownload)
        connection->send(Messages::ServiceWorkerDownloadTask::DidReceiveData { IPC::SharedBufferReference(buffer), buffer.size() }, m_fetchIdentifier);
    else
        connection->send(Messages::ServiceWorkerFetchTask::DidReceiveData { IPC::SharedBufferReference(buffer), buffer.size() }, m_fetchIdentifier);
}

void WebServiceWorkerFetchTaskClient::didFinishBlobLoading()
{
    didFinish({ });

    std::exchange(m_blobLoader, nullptr);
}

void WebServiceWorkerFetchTaskClient::didFail(const ResourceError& error)
{
    Locker lock(m_connectionLock);
    didFailInternal(error);
}

void WebServiceWorkerFetchTaskClient::didFailInternal(const ResourceError& error)
{
    RefPtr connection = m_connection;
    if (!connection)
        return;

    if (m_waitingForContinueDidReceiveResponseMessage) {
        RELEASE_LOG(ServiceWorker, "ServiceWorkerFrameLoaderClient::didFail while waiting, fetch identifier %" PRIu64, m_fetchIdentifier.toUInt64());

        m_responseData = makeUniqueRef<ResourceError>(error.isolatedCopy());
        return;
    }

    if (m_isDownload)
        connection->send(Messages::ServiceWorkerDownloadTask::DidFail { error }, m_fetchIdentifier);
    else
        connection->send(Messages::ServiceWorkerFetchTask::DidFail { error }, m_fetchIdentifier);

    cleanup();
}

void WebServiceWorkerFetchTaskClient::didFinish(const NetworkLoadMetrics& metrics)
{
    Locker lock(m_connectionLock);
    didFinishInternal(metrics);
}

void WebServiceWorkerFetchTaskClient::didFinishInternal(const NetworkLoadMetrics& metrics)
{
    RefPtr connection = m_connection;
    if (!connection)
        return;

    if (m_waitingForContinueDidReceiveResponseMessage) {
        RELEASE_LOG(ServiceWorker, "ServiceWorkerFrameLoaderClient::didFinish while waiting, fetch identifier %" PRIu64, m_fetchIdentifier.toUInt64());

        m_didFinish = true;
        m_networkLoadMetrics = metrics.isolatedCopy();
        return;
    }

    if (m_isDownload)
        connection->send(Messages::ServiceWorkerDownloadTask::DidFinish { }, m_fetchIdentifier);
    else
        connection->send(Messages::ServiceWorkerFetchTask::DidFinish { metrics }, m_fetchIdentifier);

    cleanup();
}

void WebServiceWorkerFetchTaskClient::didNotHandle()
{
    Locker lock(m_connectionLock);
    didNotHandleInternal();
}

void WebServiceWorkerFetchTaskClient::didNotHandleInternal()
{
    RefPtr connection = m_connection;
    if (!connection)
        return;

    connection->send(Messages::ServiceWorkerFetchTask::DidNotHandle { }, m_fetchIdentifier);

    cleanup();
}

void WebServiceWorkerFetchTaskClient::doCancel()
{
    Locker lock(m_connectionLock);

    ASSERT(!isMainRunLoop());
    m_connection = nullptr;
    if (m_cancelledCallback)
        m_cancelledCallback();
}

void WebServiceWorkerFetchTaskClient::convertFetchToDownload()
{
    m_isDownload = true;
    continueDidReceiveResponse();
}

void WebServiceWorkerFetchTaskClient::setCancelledCallback(Function<void()>&& callback)
{
    ASSERT(!m_cancelledCallback);
    m_cancelledCallback = WTFMove(callback);
}

void WebServiceWorkerFetchTaskClient::usePreload()
{
    Locker lock(m_connectionLock);

    RefPtr connection = m_connection;
    if (!connection)
        return;

    connection->send(Messages::ServiceWorkerFetchTask::UsePreload { }, m_fetchIdentifier);

    cleanup();
}

void WebServiceWorkerFetchTaskClient::continueDidReceiveResponse()
{
    Locker lock(m_connectionLock);

    RELEASE_LOG(ServiceWorker, "ServiceWorkerFrameLoaderClient::continueDidReceiveResponse, has connection %d, didFinish %d, response type %ld", !!m_connection, !!m_didFinish, static_cast<long>(m_responseData.index()));

    if (!m_connection)
        return;

    m_waitingForContinueDidReceiveResponseMessage = false;

    switchOn(m_responseData, [this](std::nullptr_t&) {
        assertIsHeld(m_connectionLock);
        if (m_didFinish)
            didFinishInternal(m_networkLoadMetrics);
    }, [this](const SharedBufferBuilder& buffer) {
        assertIsHeld(m_connectionLock);
        didReceiveDataInternal(buffer.copy()->makeContiguous());
        if (m_didFinish)
            didFinishInternal(m_networkLoadMetrics);
    }, [this](Ref<FormData>& formData) {
        assertIsHeld(m_connectionLock);
        didReceiveFormDataAndFinishInternal(WTFMove(formData));
    }, [this](UniqueRef<ResourceError>& error) {
        assertIsHeld(m_connectionLock);
        didFailInternal(error.get());
    });
    m_responseData = nullptr;
}

void WebServiceWorkerFetchTaskClient::cleanup()
{
    m_connection = nullptr;
    ensureOnMainRunLoop([serviceWorkerIdentifier = m_serviceWorkerIdentifier, serverConnectionIdentifier = m_serverConnectionIdentifier, fetchIdentifier = m_fetchIdentifier, needsContinueDidReceiveResponseMessage = m_needsContinueDidReceiveResponseMessage] {
        SWContextManager::singleton().removeFetch(serviceWorkerIdentifier, serverConnectionIdentifier, fetchIdentifier, needsContinueDidReceiveResponseMessage);
    });
}

void WebServiceWorkerFetchTaskClient::contextIsStopping()
{
    Locker lock(m_connectionLock);

    RefPtr connection = m_connection;
    if (!connection)
        return;

    if (!m_didSendResponse) {
        didNotHandleInternal();
        return;
    }

    if (m_didFinish) {
        ASSERT(m_needsContinueDidReceiveResponseMessage);
        return;
    }

    connection->send(Messages::ServiceWorkerFetchTask::WorkerClosed { }, m_fetchIdentifier);
    cleanup();
}

} // namespace WebKit
