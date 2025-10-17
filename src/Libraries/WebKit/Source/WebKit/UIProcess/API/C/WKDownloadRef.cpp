/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 1, 2024.
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
#include "WKDownloadRef.h"

#include "APIArray.h"
#include "APIClient.h"
#include "APIData.h"
#include "APIDownloadClient.h"
#include "APIURLRequest.h"
#include "DownloadProxy.h"
#include "WKAPICast.h"
#include "WKDownloadClient.h"
#include "WebPageProxy.h"

namespace API {
template<> struct ClientTraits<WKDownloadClientBase> {
    typedef std::tuple<WKDownloadClientV0> Versions;
};
}

using namespace WebKit;

WKTypeID WKDownloadGetTypeID()
{
    return toAPI(DownloadProxy::APIType);
}

WKURLRequestRef WKDownloadCopyRequest(WKDownloadRef download)
{
    return toAPI(&API::URLRequest::create(toImpl(download)->request()).leakRef());
}

void WKDownloadCancel(WKDownloadRef download, const void* functionContext, WKDownloadCancelCallback callback)
{
    return toImpl(download)->cancel([functionContext, callback](auto* resumeData) {
        if (callback)
            callback(toAPI(resumeData), functionContext);
    });
}

WKPageRef WKDownloadGetOriginatingPage(WKDownloadRef download)
{
    return toAPI(toImpl(download)->originatingPage());
}

bool WKDownloadGetWasUserInitiated(WKDownloadRef download)
{
    return toImpl(download)->wasUserInitiated();
}

void WKDownloadSetClient(WKDownloadRef download, WKDownloadClientBase* client)
{
    class DownloadClient : public API::Client<WKDownloadClientBase>, public API::DownloadClient {
    public:
        explicit DownloadClient(const WKDownloadClientBase* client)
        {
            initialize(client);
        }

    private:
        
        void didReceiveAuthenticationChallenge(WebKit::DownloadProxy& download, WebKit::AuthenticationChallengeProxy& challenge) override
        {
            if (!m_client.didReceiveAuthenticationChallenge) {
                challenge.listener().completeChallenge(WebKit::AuthenticationChallengeDisposition::RejectProtectionSpaceAndContinue);
                return;
            }
            m_client.didReceiveAuthenticationChallenge(toAPI(download), toAPI(challenge), m_client.base.clientInfo);
        }

        void didReceiveData(WebKit::DownloadProxy& download, uint64_t bytesWritten, uint64_t totalBytesWritten, uint64_t totalBytesExpectedToWrite) override
        {
            if (!m_client.didWriteData)
                return;
            m_client.didWriteData(toAPI(download), bytesWritten, totalBytesWritten, totalBytesExpectedToWrite, m_client.base.clientInfo);

        }

        void decideDestinationWithSuggestedFilename(WebKit::DownloadProxy& download, const WebCore::ResourceResponse& response, const WTF::String& suggestedFilename, CompletionHandler<void(WebKit::AllowOverwrite, WTF::String)>&& completionHandler) override
        {
            if (!m_client.decideDestinationWithResponse) {
                completionHandler(WebKit::AllowOverwrite::No, { });
                return;
            }
            auto destination = adoptRef(toImpl(m_client.decideDestinationWithResponse(toAPI(download), toAPI(response), toAPI(suggestedFilename.impl()), m_client.base.clientInfo)));
            if (!destination) {
                completionHandler(WebKit::AllowOverwrite::No, { });
                return;
            }
            completionHandler(WebKit::AllowOverwrite::No, destination->string());
        }

        void didFinish(WebKit::DownloadProxy& download) override
        {
            if (!m_client.didFinish)
                return;
            m_client.didFinish(toAPI(download), m_client.base.clientInfo);
        }

        void didFail(WebKit::DownloadProxy& download, const WebCore::ResourceError& error, API::Data* resumeData) override
        {
            if (!m_client.didFailWithError)
                return;
            m_client.didFailWithError(toAPI(download), toAPI(error), toAPI(resumeData), m_client.base.clientInfo);
        }

        void processDidCrash(WebKit::DownloadProxy& download) override
        {
            didFail(download, WebCore::ResourceError { WebCore::errorDomainWebKitInternal, 0, download.request().url(), "Network process crashed during download"_s }, nullptr);
        }

        void willSendRequest(WebKit::DownloadProxy& download, WebCore::ResourceRequest&& request, const WebCore::ResourceResponse& response, CompletionHandler<void(WebCore::ResourceRequest&&)>&& completionHandler) override
        {
            if (!m_client.willPerformHTTPRedirection) {
                completionHandler(WTFMove(request));
                return;
            }
            if (!m_client.willPerformHTTPRedirection(toAPI(download), toAPI(response), toAPI(request), m_client.base.clientInfo)) {
                completionHandler({ });
                return;
            }
            completionHandler(WTFMove(request));
        }
    };

    toImpl(download)->setClient(adoptRef(*new DownloadClient(client)));
}
