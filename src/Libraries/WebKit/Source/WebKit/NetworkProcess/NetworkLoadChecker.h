/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 19, 2023.
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

#include "UserContentControllerIdentifier.h"
#include "WebPageProxyIdentifier.h"
#include <WebCore/ContentSecurityPolicyResponseHeaders.h>
#include <WebCore/CrossOriginEmbedderPolicy.h>
#include <WebCore/FetchOptions.h>
#include <WebCore/NetworkLoadInformation.h>
#include <pal/SessionID.h>
#include <variant>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class ContentSecurityPolicy;
class OriginAccessPatterns;
struct ContentRuleListResults;
struct ContentSecurityPolicyClient;
class ResourceError;
class SecurityOrigin;
enum class AdvancedPrivacyProtections : uint16_t;
enum class PreflightPolicy : uint8_t;
enum class StoredCredentialsPolicy : uint8_t;
}

namespace WebKit {

class NetworkCORSPreflightChecker;
class NetworkProcess;
class NetworkResourceLoader;
class NetworkSchemeRegistry;

using DocumentURL = URL;

class NetworkLoadChecker : public RefCountedAndCanMakeWeakPtr<NetworkLoadChecker> {
    WTF_MAKE_TZONE_ALLOCATED(NetworkLoadChecker);
public:
    enum class LoadType : bool { MainFrame, Other };

    static Ref<NetworkLoadChecker> create(NetworkProcess& networkProcess, NetworkResourceLoader* networkResourceLoader, NetworkSchemeRegistry* schemeRegistry,
        WebCore::FetchOptions&& options, PAL::SessionID sessionID, std::optional<WebPageProxyIdentifier> webPageProxyID, WebCore::HTTPHeaderMap&& originalRequestHeaders,
        URL&& url, DocumentURL&& documentURL, RefPtr<WebCore::SecurityOrigin>&& sourceOrigin, RefPtr<WebCore::SecurityOrigin>&& topOrigin,
        RefPtr<WebCore::SecurityOrigin>&& parentOrigin, WebCore::PreflightPolicy preflightPolicy, String&& referrer, bool allowPrivacyProxy,
        OptionSet<WebCore::AdvancedPrivacyProtections> advancedPrivacyProtections, bool shouldCaptureExtraNetworkLoadMetrics = false, LoadType requestLoadType = LoadType::Other)
    {
        return adoptRef(*new NetworkLoadChecker(networkProcess, networkResourceLoader, schemeRegistry,
            WTFMove(options), sessionID, webPageProxyID, WTFMove(originalRequestHeaders),
            WTFMove(url), WTFMove(documentURL), WTFMove(sourceOrigin), WTFMove(topOrigin),
            WTFMove(parentOrigin), preflightPolicy, WTFMove(referrer), allowPrivacyProxy,
            advancedPrivacyProtections, shouldCaptureExtraNetworkLoadMetrics, requestLoadType));
    }

    ~NetworkLoadChecker();

    struct RedirectionTriplet {
        WebCore::ResourceRequest request;
        WebCore::ResourceRequest redirectRequest;
        WebCore::ResourceResponse redirectResponse;
    };

    using RequestOrRedirectionTripletOrError = std::variant<WebCore::ResourceRequest, RedirectionTriplet, WebCore::ResourceError>;
    using ValidationHandler = CompletionHandler<void(RequestOrRedirectionTripletOrError&&)>;
    void check(WebCore::ResourceRequest&&, WebCore::ContentSecurityPolicyClient*, ValidationHandler&&);

    using RedirectionRequestOrError = Expected<RedirectionTriplet, WebCore::ResourceError>;
    using RedirectionValidationHandler = CompletionHandler<void(RedirectionRequestOrError&&)>;
    void checkRedirection(WebCore::ResourceRequest&& request, WebCore::ResourceRequest&& redirectRequest, WebCore::ResourceResponse&& redirectResponse, WebCore::ContentSecurityPolicyClient*, RedirectionValidationHandler&&);

    WebCore::ResourceError validateResponse(const WebCore::ResourceRequest&, WebCore::ResourceResponse&);

    void setCSPResponseHeaders(WebCore::ContentSecurityPolicyResponseHeaders&& headers) { m_cspResponseHeaders = WTFMove(headers); }
    void setParentCrossOriginEmbedderPolicy(const WebCore::CrossOriginEmbedderPolicy& parentCrossOriginEmbedderPolicy) { m_parentCrossOriginEmbedderPolicy = parentCrossOriginEmbedderPolicy; }
    void setCrossOriginEmbedderPolicy(const WebCore::CrossOriginEmbedderPolicy& crossOriginEmbedderPolicy) { m_crossOriginEmbedderPolicy = crossOriginEmbedderPolicy; }
#if ENABLE(CONTENT_EXTENSIONS)
    void setContentExtensionController(URL&& mainDocumentURL, URL&& frameURL, std::optional<UserContentControllerIdentifier> identifier)
    {
        m_mainDocumentURL = WTFMove(mainDocumentURL);
        m_frameURL = WTFMove(frameURL);
        m_userContentControllerIdentifier = identifier;
    }
#endif

    NetworkProcess& networkProcess() { return m_networkProcess; }
    Ref<NetworkProcess> protectedNetworkProcess();

    const URL& url() const { return m_url; }
    WebCore::StoredCredentialsPolicy storedCredentialsPolicy() const { return m_storedCredentialsPolicy; }

    WebCore::NetworkLoadInformation takeNetworkLoadInformation() { return WTFMove(m_loadInformation); }
    void storeRedirectionIfNeeded(const WebCore::ResourceRequest&, const WebCore::ResourceResponse&);

    void enableContentExtensionsCheck() { m_checkContentExtensions = true; }

    RefPtr<WebCore::SecurityOrigin> origin() const { return m_origin; }
    RefPtr<WebCore::SecurityOrigin> topOrigin() const { return m_topOrigin; }

    const WebCore::FetchOptions& options() const { return m_options; }

    bool timingAllowFailedFlag() const { return m_timingAllowFailedFlag; }

private:
    NetworkLoadChecker(NetworkProcess&, NetworkResourceLoader*, NetworkSchemeRegistry*, WebCore::FetchOptions&&, PAL::SessionID, std::optional<WebPageProxyIdentifier>, WebCore::HTTPHeaderMap&&, URL&&, DocumentURL&&,  RefPtr<WebCore::SecurityOrigin>&&, RefPtr<WebCore::SecurityOrigin>&& topOrigin, RefPtr<WebCore::SecurityOrigin>&& parentOrigin, WebCore::PreflightPolicy, String&& referrer, bool allowPrivacyProxy, OptionSet<WebCore::AdvancedPrivacyProtections>, bool shouldCaptureExtraNetworkLoadMetrics, LoadType requestLoadType);

    WebCore::ContentSecurityPolicy* contentSecurityPolicy();
    const WebCore::OriginAccessPatterns& originAccessPatterns() const;
    bool isSameOrigin(const URL&, const WebCore::SecurityOrigin*) const;
    bool isChecking() const { return !!m_corsPreflightChecker; }
    bool isRedirected() const { return m_redirectCount; }

    void checkRequest(WebCore::ResourceRequest&&, WebCore::ContentSecurityPolicyClient*, ValidationHandler&&);

    bool isAllowedByContentSecurityPolicy(const WebCore::ResourceRequest&, WebCore::ContentSecurityPolicyClient*);

    void continueCheckingRequest(WebCore::ResourceRequest&&, ValidationHandler&&);
    void continueCheckingRequestOrDoSyntheticRedirect(WebCore::ResourceRequest&& originalRequest, WebCore::ResourceRequest&& currentRequest, ValidationHandler&&);

    bool doesNotNeedCORSCheck(const URL&) const;
    void checkCORSRequest(WebCore::ResourceRequest&&, ValidationHandler&&);
    void checkCORSRedirectedRequest(WebCore::ResourceRequest&&, ValidationHandler&&);
    void checkCORSRequestWithPreflight(WebCore::ResourceRequest&&, ValidationHandler&&);

    RequestOrRedirectionTripletOrError accessControlErrorForValidationHandler(String&&);

#if ENABLE(CONTENT_EXTENSIONS)
    struct ContentExtensionResult {
        WebCore::ResourceRequest request;
        const WebCore::ContentRuleListResults& results;
    };
    using ContentExtensionResultOrError = Expected<ContentExtensionResult, WebCore::ResourceError>;
    using ContentExtensionCallback = CompletionHandler<void(ContentExtensionResultOrError&&)>;
    void processContentRuleListsForLoad(WebCore::ResourceRequest&&, ContentExtensionCallback&&);
#endif

    RefPtr<WebCore::SecurityOrigin> parentOrigin() const { return m_parentOrigin; }

    bool checkTAO(const WebCore::ResourceResponse&);

    WebCore::FetchOptions m_options;
    WebCore::StoredCredentialsPolicy m_storedCredentialsPolicy;
    bool m_allowPrivacyProxy;
    OptionSet<WebCore::AdvancedPrivacyProtections> m_advancedPrivacyProtections;
    PAL::SessionID m_sessionID;
    Ref<NetworkProcess> m_networkProcess;
    Markable<WebPageProxyIdentifier> m_webPageProxyID;
    WebCore::HTTPHeaderMap m_originalRequestHeaders; // Needed for CORS checks.
    WebCore::HTTPHeaderMap m_firstRequestHeaders; // Needed for CORS checks.
    URL m_url;
    DocumentURL m_documentURL;
    RefPtr<WebCore::SecurityOrigin> m_origin;
    RefPtr<WebCore::SecurityOrigin> m_topOrigin;
    RefPtr<WebCore::SecurityOrigin> m_parentOrigin;
    std::optional<WebCore::ContentSecurityPolicyResponseHeaders> m_cspResponseHeaders;
    WebCore::CrossOriginEmbedderPolicy m_parentCrossOriginEmbedderPolicy;
    WebCore::CrossOriginEmbedderPolicy m_crossOriginEmbedderPolicy;
#if ENABLE(CONTENT_EXTENSIONS)
    URL m_mainDocumentURL;
    URL m_frameURL;
    std::optional<UserContentControllerIdentifier> m_userContentControllerIdentifier;
#endif

    RefPtr<NetworkCORSPreflightChecker> m_corsPreflightChecker;
    bool m_isSameOriginRequest { true };
    bool m_isSimpleRequest { true };
    std::unique_ptr<WebCore::ContentSecurityPolicy> m_contentSecurityPolicy;
    size_t m_redirectCount { 0 };
    URL m_previousURL;
    WebCore::PreflightPolicy m_preflightPolicy;
    String m_referrer;
    bool m_checkContentExtensions { false };
    bool m_shouldCaptureExtraNetworkLoadMetrics { false };

    WebCore::NetworkLoadInformation m_loadInformation;

    LoadType m_requestLoadType;
    RefPtr<NetworkSchemeRegistry> m_schemeRegistry;
    WeakPtr<NetworkResourceLoader> m_networkResourceLoader;

    bool m_timingAllowFailedFlag { false };
};

}
