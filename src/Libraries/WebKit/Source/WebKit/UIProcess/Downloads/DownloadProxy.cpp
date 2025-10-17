/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 28, 2024.
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
#include "DownloadProxy.h"

#include "APIData.h"
#include "APIDownloadClient.h"
#include "APIFrameInfo.h"
#include "AuthenticationChallengeProxy.h"
#include "DownloadProxyMap.h"
#include "FrameInfoData.h"
#include "NetworkProcessMessages.h"
#include "NetworkProcessProxy.h"
#include "ProcessAssertion.h"
#include "WebPageProxy.h"
#include "WebProcessMessages.h"
#include "WebProtectionSpace.h"
#include <WebCore/MIMETypeRegistry.h>
#include <WebCore/ResourceResponseBase.h>
#include <wtf/FileSystem.h>
#include <wtf/text/CString.h>
#include <wtf/text/WTFString.h>

#if PLATFORM(MAC)
#include <pal/spi/mac/QuarantineSPI.h>
#endif

#if HAVE(SEC_KEY_PROXY)
#include "SecKeyProxyStore.h"
#endif

namespace WebKit {
using namespace WebCore;

DownloadProxy::DownloadProxy(DownloadProxyMap& downloadProxyMap, WebsiteDataStore& dataStore, API::DownloadClient& client, const ResourceRequest& resourceRequest, const FrameInfoData& frameInfoData, WebPageProxy* originatingPage)
    : m_downloadProxyMap(downloadProxyMap)
    , m_dataStore(&dataStore)
    , m_client(client)
    , m_downloadID(DownloadID::generate())
    , m_request(resourceRequest)
    , m_originatingPage(originatingPage)
    , m_frameInfo(API::FrameInfo::create(FrameInfoData { frameInfoData }, originatingPage))
#if HAVE(MODERN_DOWNLOADPROGRESS)
    , m_assertion(ProcessAssertion::create(getCurrentProcessID(), "WebKit DownloadProxy DecideDestination"_s, ProcessAssertionType::FinishTaskInterruptable))
#endif
{
}

DownloadProxy::~DownloadProxy()
{
    if (m_didStartCallback)
        m_didStartCallback(nullptr);
}

static RefPtr<API::Data> createData(std::span<const uint8_t> data)
{
    if (data.empty())
        return nullptr;
    return API::Data::create(data);
}

void DownloadProxy::cancel(CompletionHandler<void(API::Data*)>&& completionHandler)
{
    m_downloadIsCancelled = true;
    if (m_dataStore) {
        protectedDataStore()->protectedNetworkProcess()->sendWithAsyncReply(Messages::NetworkProcess::CancelDownload(m_downloadID), [weakThis = WeakPtr { *this }, completionHandler = WTFMove(completionHandler)] (std::span<const uint8_t> resumeData) mutable {
            RefPtr protectedThis = weakThis.get();
            if (!protectedThis)
                return completionHandler(nullptr);
            protectedThis->m_legacyResumeData = createData(resumeData);
            completionHandler(protectedThis->m_legacyResumeData.get());
            if (RefPtr downloadProxyMap = protectedThis->m_downloadProxyMap.get())
                downloadProxyMap->downloadFinished(*protectedThis);
        });
    } else
        completionHandler(nullptr);
}

void DownloadProxy::invalidate()
{
    ASSERT(m_dataStore);
    m_dataStore = nullptr;
}

void DownloadProxy::processDidClose()
{
    m_client->processDidCrash(*this);
}

WebPageProxy* DownloadProxy::originatingPage() const
{
    return m_originatingPage.get();
}

void DownloadProxy::didStart(const ResourceRequest& request, const String& suggestedFilename)
{
    m_request = request;
    m_suggestedFilename = suggestedFilename;

    if (m_redirectChain.isEmpty() || m_redirectChain.last() != request.url())
        m_redirectChain.append(request.url());

    if (m_didStartCallback)
        m_didStartCallback(this);
    m_client->legacyDidStart(*this);
}

void DownloadProxy::didReceiveAuthenticationChallenge(AuthenticationChallenge&& authenticationChallenge, AuthenticationChallengeIdentifier challengeID)
{
    RefPtr dataStore = m_dataStore;
    if (!dataStore)
        return;

    auto authenticationChallengeProxy = AuthenticationChallengeProxy::create(WTFMove(authenticationChallenge), challengeID, dataStore->networkProcess().connection(), nullptr);
    protectedClient()->didReceiveAuthenticationChallenge(*this, authenticationChallengeProxy.get());
}

void DownloadProxy::willSendRequest(ResourceRequest&& proposedRequest, const ResourceResponse& redirectResponse, CompletionHandler<void(ResourceRequest&&)>&& completionHandler)
{
    protectedClient()->willSendRequest(*this, WTFMove(proposedRequest), redirectResponse, [this, protectedThis = Ref { *this }, completionHandler = WTFMove(completionHandler)] (ResourceRequest&& newRequest) mutable {
        m_redirectChain.append(newRequest.url());
        completionHandler(WTFMove(newRequest));
    });
}

void DownloadProxy::didReceiveData(uint64_t bytesWritten, uint64_t totalBytesWritten, uint64_t totalBytesExpectedToWrite)
{
    m_client->didReceiveData(*this, bytesWritten, totalBytesWritten, totalBytesExpectedToWrite);
}

void DownloadProxy::decideDestinationWithSuggestedFilename(const WebCore::ResourceResponse& response, String&& suggestedFilename, DecideDestinationCallback&& completionHandler)
{
    RELEASE_LOG_INFO_IF(!response.expectedContentLength(), Network, "DownloadProxy::decideDestinationWithSuggestedFilename expectedContentLength is null");

    // As per https://html.spec.whatwg.org/#as-a-download (step 2), the filename from the Content-Disposition header
    // should override the suggested filename from the download attribute.
    if (response.isAttachmentWithFilename() || (suggestedFilename.isEmpty() && m_suggestedFilename.isEmpty()))
        suggestedFilename = response.suggestedFilename();
    else if (!m_suggestedFilename.isEmpty())
        suggestedFilename = m_suggestedFilename;
    suggestedFilename = MIMETypeRegistry::appendFileExtensionIfNecessary(suggestedFilename, response.mimeType());

    protectedClient()->decideDestinationWithSuggestedFilename(*this, response, ResourceResponseBase::sanitizeSuggestedFilename(suggestedFilename), [this, protectedThis = Ref { *this }, completionHandler = WTFMove(completionHandler)] (AllowOverwrite allowOverwrite, String destination) mutable {
        SandboxExtension::Handle sandboxExtensionHandle;
        if (!destination.isNull()) {
            if (auto handle = SandboxExtension::createHandle(destination, SandboxExtension::Type::ReadWrite))
                sandboxExtensionHandle = WTFMove(*handle);
        }

        setDestinationFilename(destination);

        protectedClient()->decidePlaceholderPolicy(*this, [completionHandler = WTFMove(completionHandler), destination = WTFMove(destination), sandboxExtensionHandle = WTFMove(sandboxExtensionHandle), allowOverwrite] (WebKit::UseDownloadPlaceholder usePlaceholder, const URL& url) mutable {

            SandboxExtension::Handle placeHolderSandboxExtensionHandle;
            Vector<uint8_t> bookmarkData;
            Vector<uint8_t> activityTokenData;
#if HAVE(MODERN_DOWNLOADPROGRESS)
            bookmarkData = bookmarkDataForURL(url);
            activityTokenData = activityAccessToken();
#else
            if (auto handle = SandboxExtension::createHandle(url.fileSystemPath(), SandboxExtension::Type::ReadWrite))
                placeHolderSandboxExtensionHandle = WTFMove(*handle);
#endif
            completionHandler(destination, WTFMove(sandboxExtensionHandle), allowOverwrite, usePlaceholder, url, WTFMove(placeHolderSandboxExtensionHandle), bookmarkData.span(), activityTokenData.span());
        });
    });
}

void DownloadProxy::didCreateDestination(const String& path)
{
    m_client->didCreateDestination(*this, path);
}

#if PLATFORM(MAC)
void DownloadProxy::updateQuarantinePropertiesIfPossible()
{
    auto fileURL = URL::fileURLWithFileSystemPath(m_destinationFilename);
    auto path = fileURL.fileSystemPath().utf8();

    auto file = std::unique_ptr<_qtn_file, QuarantineFileDeleter>(qtn_file_alloc());
    if (!file)
        return;

    auto error = qtn_file_init_with_path(file.get(), path.data());
    if (error)
        return;

    uint32_t flags = qtn_file_get_flags(file.get());
    ASSERT_WITH_MESSAGE(flags & QTN_FLAG_HARD, "Downloaded files written by the sandboxed network process should have QTN_FLAG_HARD");
    flags &= ~QTN_FLAG_HARD;
    error = qtn_file_set_flags(file.get(), flags);
    if (error)
        return;

    qtn_file_apply_to_path(file.get(), path.data());
}
#endif

void DownloadProxy::didFinish()
{
#if PLATFORM(MAC)
    updateQuarantinePropertiesIfPossible();
#endif
    m_client->didFinish(*this);
    if (m_downloadIsCancelled)
        return;

    // This can cause the DownloadProxy object to be deleted.
    if (RefPtr downloadProxyMap = m_downloadProxyMap.get())
        downloadProxyMap->downloadFinished(*this);
}

void DownloadProxy::didFail(const ResourceError& error, std::span<const uint8_t> resumeData)
{
    if (m_downloadIsCancelled)
        return;

    m_legacyResumeData = createData(resumeData);

    m_client->didFail(*this, error, m_legacyResumeData.get());

    // This can cause the DownloadProxy object to be deleted.
    if (RefPtr downloadProxyMap = m_downloadProxyMap.get())
        downloadProxyMap->downloadFinished(*this);
}

void DownloadProxy::setClient(Ref<API::DownloadClient>&& client)
{
    m_client = WTFMove(client);
}

Ref<API::DownloadClient> DownloadProxy::protectedClient() const
{
    return m_client;
}

} // namespace WebKit

