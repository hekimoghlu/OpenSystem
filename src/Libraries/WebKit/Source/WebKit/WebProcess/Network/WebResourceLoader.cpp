/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 2, 2022.
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
#include "WebResourceLoader.h"

#include "FormDataReference.h"
#include "Logging.h"
#include "MessageSenderInlines.h"
#include "NetworkProcessConnection.h"
#include "NetworkResourceLoaderMessages.h"
#include "PrivateRelayed.h"
#include "SharedBufferReference.h"
#include "WebErrors.h"
#include "WebFrame.h"
#include "WebLoaderStrategy.h"
#include "WebLocalFrameLoaderClient.h"
#include "WebPage.h"
#include "WebProcess.h"
#include "WebURLSchemeHandlerProxy.h"
#include <WebCore/CertificateInfo.h>
#include <WebCore/DiagnosticLoggingClient.h>
#include <WebCore/DiagnosticLoggingKeys.h>
#include <WebCore/DocumentLoader.h>
#include <WebCore/FrameLoader.h>
#include <WebCore/InspectorInstrumentationWebKit.h>
#include <WebCore/LocalFrame.h>
#include <WebCore/LocalFrameLoaderClient.h>
#include <WebCore/NetworkLoadMetrics.h>
#include <WebCore/Page.h>
#include <WebCore/ResourceError.h>
#include <WebCore/ResourceLoader.h>
#include <WebCore/SubresourceLoader.h>
#include <WebCore/SubstituteData.h>
#include <wtf/CompletionHandler.h>
#include <wtf/text/MakeString.h>

#define WEBRESOURCELOADER_RELEASE_LOG(fmt, ...) RELEASE_LOG_FORWARDABLE(Network, fmt, m_trackingParameters ? m_trackingParameters->pageID.toUInt64() : 0, m_trackingParameters ? m_trackingParameters->frameID.object().toUInt64() : 0, m_trackingParameters ? m_trackingParameters->resourceID.toUInt64() : 0, ##__VA_ARGS__)

namespace WebKit {
using namespace WebCore;

Ref<WebResourceLoader> WebResourceLoader::create(Ref<ResourceLoader>&& coreLoader, const std::optional<TrackingParameters>& trackingParameters)
{
    return adoptRef(*new WebResourceLoader(WTFMove(coreLoader), trackingParameters));
}

WebResourceLoader::WebResourceLoader(Ref<WebCore::ResourceLoader>&& coreLoader, const std::optional<TrackingParameters>& trackingParameters)
    : m_coreLoader(WTFMove(coreLoader))
    , m_trackingParameters(trackingParameters)
    , m_loadStart(MonotonicTime::now())
{
    WEBRESOURCELOADER_RELEASE_LOG(WEBRESOURCELOADER_CONSTRUCTOR);
}

WebResourceLoader::~WebResourceLoader()
{
}

IPC::Connection* WebResourceLoader::messageSenderConnection() const
{
    return &WebProcess::singleton().ensureNetworkProcessConnection().connection();
}

uint64_t WebResourceLoader::messageSenderDestinationID() const
{
    RELEASE_ASSERT(RunLoop::isMain());
    return m_coreLoader->identifier()->toUInt64();
}

void WebResourceLoader::detachFromCoreLoader()
{
    RELEASE_ASSERT(RunLoop::isMain());
    m_coreLoader = nullptr;
}

MainFrameMainResource WebResourceLoader::mainFrameMainResource() const
{
    RefPtr frame = m_coreLoader->frame();
    if (!frame || !frame->isMainFrame())
        return MainFrameMainResource::No;

    RefPtr frameLoader = m_coreLoader->frameLoader();
    if (!frameLoader)
        return MainFrameMainResource::No;

    if (!frameLoader->notifier().isInitialRequestIdentifier(*m_coreLoader->identifier()))
        return MainFrameMainResource::No;

    return MainFrameMainResource::Yes;
}

void WebResourceLoader::willSendRequest(ResourceRequest&& proposedRequest, IPC::FormDataReference&& proposedRequestBody, ResourceResponse&& redirectResponse, CompletionHandler<void(ResourceRequest&&, bool)>&& completionHandler)
{
    Ref<WebResourceLoader> protectedThis(*this);

    // Make the request whole again as we do not normally encode the request's body when sending it over IPC, for performance reasons.
    proposedRequest.setHTTPBody(proposedRequestBody.takeData());

    LOG(Network, "(WebProcess) WebResourceLoader::willSendRequest to '%s'", proposedRequest.url().string().latin1().data());
    WEBRESOURCELOADER_RELEASE_LOG(WEBRESOURCELOADER_WILLSENDREQUEST);
    
    if (RefPtr frame = m_coreLoader->frame()) {
        if (RefPtr page = frame->page()) {
            if (!page->allowsLoadFromURL(proposedRequest.url(), mainFrameMainResource()))
                proposedRequest = { };
        }
    }

    m_coreLoader->willSendRequest(WTFMove(proposedRequest), redirectResponse, [this, protectedThis = WTFMove(protectedThis), completionHandler = WTFMove(completionHandler)] (ResourceRequest&& request) mutable {
        if (!m_coreLoader || !m_coreLoader->identifier()) {
            WEBRESOURCELOADER_RELEASE_LOG(WEBRESOURCELOADER_WILLSENDREQUEST_NO_CORELOADER);
            return completionHandler({ }, false);
        }

        WEBRESOURCELOADER_RELEASE_LOG(WEBRESOURCELOADER_WILLSENDREQUEST_CONTINUE);
        completionHandler(WTFMove(request), m_coreLoader->isAllowedToAskUserForCredentials());
    });
}

void WebResourceLoader::didSendData(uint64_t bytesSent, uint64_t totalBytesToBeSent)
{
    m_coreLoader->didSendData(bytesSent, totalBytesToBeSent);
}

void WebResourceLoader::didReceiveResponse(ResourceResponse&& response, PrivateRelayed privateRelayed, bool needsContinueDidReceiveResponseMessage, std::optional<NetworkLoadMetrics>&& metrics)
{
    LOG(Network, "(WebProcess) WebResourceLoader::didReceiveResponse for '%s'. Status %d.", m_coreLoader->url().string().latin1().data(), response.httpStatusCode());
    WEBRESOURCELOADER_RELEASE_LOG(WEBRESOURCELOADER_DIDRECEIVERESPONSE, response.httpStatusCode());

    Ref<WebResourceLoader> protectedThis(*this);

    if (metrics) {
        metrics->workerStart = m_workerStart;
        response.setDeprecatedNetworkLoadMetrics(Box<NetworkLoadMetrics>::create(WTFMove(*metrics)));
    }

    if (privateRelayed == PrivateRelayed::Yes && mainFrameMainResource() == MainFrameMainResource::Yes)
        WebProcess::singleton().setHadMainFrameMainResourcePrivateRelayed();

    CompletionHandler<void()> policyDecisionCompletionHandler;
    if (needsContinueDidReceiveResponseMessage) {
#if ASSERT_ENABLED
        m_isProcessingNetworkResponse = true;
#endif
        policyDecisionCompletionHandler = [this, protectedThis = WTFMove(protectedThis)] {
#if ASSERT_ENABLED
            m_isProcessingNetworkResponse = false;
#endif
            // If m_coreLoader becomes null as a result of the didReceiveResponse callback, we can't use the send function().
            if (m_coreLoader && m_coreLoader->identifier())
                send(Messages::NetworkResourceLoader::ContinueDidReceiveResponse());
            else
                WEBRESOURCELOADER_RELEASE_LOG(WEBRESOURCELOADER_DIDRECEIVERESPONSE_NOT_CONTINUING_LOAD);
        };
    }

    if (InspectorInstrumentationWebKit::shouldInterceptResponse(m_coreLoader->frame(), response)) {
        auto interceptedRequestIdentifier = *m_coreLoader->identifier();
        m_interceptController.beginInterceptingResponse(interceptedRequestIdentifier);
        InspectorInstrumentationWebKit::interceptResponse(m_coreLoader->frame(), response, interceptedRequestIdentifier, [this, protectedThis = Ref { *this }, interceptedRequestIdentifier, policyDecisionCompletionHandler = WTFMove(policyDecisionCompletionHandler)](const ResourceResponse& inspectorResponse, RefPtr<FragmentedSharedBuffer> overrideData) mutable {
            if (!m_coreLoader || !m_coreLoader->identifier()) {
                WEBRESOURCELOADER_RELEASE_LOG(WEBRESOURCELOADER_DIDRECEIVERESPONSE_NOT_CONTINUING_INTERCEPT_LOAD);
                m_interceptController.continueResponse(interceptedRequestIdentifier);
                return;
            }

            m_coreLoader->didReceiveResponse(inspectorResponse, [this, protectedThis = WTFMove(protectedThis), interceptedRequestIdentifier, policyDecisionCompletionHandler = WTFMove(policyDecisionCompletionHandler), overrideData = WTFMove(overrideData)]() mutable {
                if (policyDecisionCompletionHandler)
                    policyDecisionCompletionHandler();

                if (!m_coreLoader || !m_coreLoader->identifier()) {
                    m_interceptController.continueResponse(interceptedRequestIdentifier);
                    return;
                }

                RefPtr<WebCore::ResourceLoader> protectedCoreLoader = m_coreLoader;
                if (!overrideData)
                    m_interceptController.continueResponse(interceptedRequestIdentifier);
                else {
                    m_interceptController.interceptedResponse(interceptedRequestIdentifier);
                    if (unsigned bufferSize = overrideData->size())
                        protectedCoreLoader->didReceiveBuffer(overrideData.releaseNonNull(), bufferSize, DataPayloadWholeResource);
                    WebCore::NetworkLoadMetrics emptyMetrics;
                    protectedCoreLoader->didFinishLoading(emptyMetrics);
                }
            });
        });
        return;
    }

    m_coreLoader->didReceiveResponse(response, WTFMove(policyDecisionCompletionHandler));
}

void WebResourceLoader::didReceiveData(IPC::SharedBufferReference&& data, uint64_t encodedDataLength)
{
    LOG(Network, "(WebProcess) WebResourceLoader::didReceiveData of size %lu for '%s'", data.size(), m_coreLoader->url().string().latin1().data());
    ASSERT_WITH_MESSAGE(!m_isProcessingNetworkResponse, "Network process should not send data until we've validated the response");

    if (UNLIKELY(m_interceptController.isIntercepting(*m_coreLoader->identifier()))) {
        m_interceptController.defer(*m_coreLoader->identifier(), [this, protectedThis = Ref { *this }, buffer = WTFMove(data), encodedDataLength]() mutable {
            if (m_coreLoader)
                didReceiveData(WTFMove(buffer), encodedDataLength);
        });
        return;
    }

    if (!m_numBytesReceived)
        WEBRESOURCELOADER_RELEASE_LOG(WEBRESOURCELOADER_DIDRECEIVEDATA);
    m_numBytesReceived += data.size();

    m_coreLoader->didReceiveData(data.isNull() ? SharedBuffer::create() : data.unsafeBuffer().releaseNonNull(), encodedDataLength, DataPayloadBytes);
}

void WebResourceLoader::didFinishResourceLoad(NetworkLoadMetrics&& networkLoadMetrics)
{
    LOG(Network, "(WebProcess) WebResourceLoader::didFinishResourceLoad for '%s'", m_coreLoader->url().string().latin1().data());
    WEBRESOURCELOADER_RELEASE_LOG(WEBRESOURCELOADER_DIDFINISHRESOURCELOAD, m_numBytesReceived);

    if (UNLIKELY(m_interceptController.isIntercepting(*m_coreLoader->identifier()))) {
        m_interceptController.defer(*m_coreLoader->identifier(), [this, protectedThis = Ref { *this }, networkLoadMetrics = WTFMove(networkLoadMetrics)]() mutable {
            if (m_coreLoader)
                didFinishResourceLoad(WTFMove(networkLoadMetrics));
        });
        return;
    }

    networkLoadMetrics.workerStart = m_workerStart;

    ASSERT_WITH_MESSAGE(!m_isProcessingNetworkResponse, "Load should not be able to finish before we've validated the response");
    m_coreLoader->didFinishLoading(networkLoadMetrics);
}

void WebResourceLoader::didFailServiceWorkerLoad(const ResourceError& error)
{
    if (RefPtr document = m_coreLoader->frame() ? m_coreLoader->frame()->document() : nullptr) {
        if (m_coreLoader->options().destination != FetchOptions::Destination::EmptyString || error.isGeneral())
            document->addConsoleMessage(MessageSource::JS, MessageLevel::Error, error.localizedDescription());
        if (m_coreLoader->options().destination != FetchOptions::Destination::EmptyString)
            document->addConsoleMessage(MessageSource::JS, MessageLevel::Error, makeString("Cannot load "_s, error.failingURL().string(), '.'));
    }

    didFailResourceLoad(error);
}

void WebResourceLoader::serviceWorkerDidNotHandle()
{
    WEBRESOURCELOADER_RELEASE_LOG(WEBRESOURCELOADER_SERVICEWORKERDIDNOTHANDLE);

    ASSERT(m_coreLoader->options().serviceWorkersMode == ServiceWorkersMode::Only);
    auto error = internalError(m_coreLoader->request().url());
    error.setType(ResourceError::Type::Cancellation);
    m_coreLoader->didFail(error);
}

void WebResourceLoader::didFailResourceLoad(const ResourceError& error)
{
    LOG(Network, "(WebProcess) WebResourceLoader::didFailResourceLoad for '%s'", m_coreLoader->url().string().latin1().data());
    WEBRESOURCELOADER_RELEASE_LOG(WEBRESOURCELOADER_DIDFAILRESOURCELOAD);

    if (UNLIKELY(m_interceptController.isIntercepting(*m_coreLoader->identifier()))) {
        m_interceptController.defer(*m_coreLoader->identifier(), [this, protectedThis = Ref { *this }, error]() mutable {
            if (m_coreLoader)
                didFailResourceLoad(error);
        });
        return;
    }

    ASSERT_WITH_MESSAGE(!m_isProcessingNetworkResponse, "Load should not be able to finish before we've validated the response");

    m_coreLoader->didFail(error);
}

void WebResourceLoader::didBlockAuthenticationChallenge()
{
    LOG(Network, "(WebProcess) WebResourceLoader::didBlockAuthenticationChallenge for '%s'", m_coreLoader->url().string().latin1().data());
    WEBRESOURCELOADER_RELEASE_LOG(WEBRESOURCELOADER_DIDBLOCKAUTHENTICATIONCHALLENGE);

    m_coreLoader->didBlockAuthenticationChallenge();
}

void WebResourceLoader::stopLoadingAfterXFrameOptionsOrContentSecurityPolicyDenied(const ResourceResponse& response)
{
    LOG(Network, "(WebProcess) WebResourceLoader::stopLoadingAfterXFrameOptionsOrContentSecurityPolicyDenied for '%s'", m_coreLoader->url().string().latin1().data());
    WEBRESOURCELOADER_RELEASE_LOG(WEBRESOURCELOADER_STOPLOADINGAFTERSECURITYPOLICYDENIED);

    m_coreLoader->protectedDocumentLoader()->stopLoadingAfterXFrameOptionsOrContentSecurityPolicyDenied(*m_coreLoader->identifier(), response);
}

#if ENABLE(SHAREABLE_RESOURCE)
void WebResourceLoader::didReceiveResource(ShareableResource::Handle&& handle)
{
    LOG(Network, "(WebProcess) WebResourceLoader::didReceiveResource for '%s'", m_coreLoader->url().string().latin1().data());
    WEBRESOURCELOADER_RELEASE_LOG(WEBRESOURCELOADER_DIDRECEIVERESOURCE);

    RefPtr<SharedBuffer> buffer = WTFMove(handle).tryWrapInSharedBuffer();

    if (!buffer) {
        LOG_ERROR("Unable to create buffer from ShareableResource sent from the network process.");
        WEBRESOURCELOADER_RELEASE_LOG(WEBRESOURCELOADER_DIDRECEIVERESOURCE_UNABLE_TO_CREATE_FRAGMENTEDSHAREDBUFFER);
        if (RefPtr frame = m_coreLoader->frame()) {
            if (RefPtr page = frame->page())
                page->diagnosticLoggingClient().logDiagnosticMessage(WebCore::DiagnosticLoggingKeys::internalErrorKey(), WebCore::DiagnosticLoggingKeys::createSharedBufferFailedKey(), WebCore::ShouldSample::No);
        }
        m_coreLoader->didFail(internalError(m_coreLoader->request().url()));
        return;
    }

    Ref<WebResourceLoader> protect(*this);

    // Only send data to the didReceiveData callback if it exists.
    if (unsigned bufferSize = buffer->size())
        m_coreLoader->didReceiveData(buffer.releaseNonNull(), bufferSize, DataPayloadWholeResource);

    if (!m_coreLoader)
        return;

    NetworkLoadMetrics emptyMetrics;
    m_coreLoader->didFinishLoading(emptyMetrics);
}
#endif

#if ENABLE(CONTENT_FILTERING)
void WebResourceLoader::contentFilterDidBlockLoad(const WebCore::ContentFilterUnblockHandler& unblockHandler, String&& unblockRequestDeniedScript, const ResourceError& error, const URL& blockedPageURL,  WebCore::SubstituteData&& substituteData)
{
    if (!m_coreLoader || !m_coreLoader->documentLoader())
        return;
    RefPtr documentLoader = m_coreLoader->documentLoader();
    documentLoader->setBlockedPageURL(blockedPageURL);
    documentLoader->setSubstituteDataFromContentFilter(WTFMove(substituteData));
    documentLoader->handleContentFilterDidBlock(unblockHandler, WTFMove(unblockRequestDeniedScript));
    documentLoader->cancelMainResourceLoad(error);
}
#endif // ENABLE(CONTENT_FILTERING)

} // namespace WebKit

#undef WEBRESOURCELOADER_RELEASE_LOG
