/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 31, 2022.
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

#include "NetworkLoadParameters.h"
#include "PolicyDecision.h"
#include "SandboxExtension.h"
#include "UserContentControllerIdentifier.h"
#include <WebCore/ContentSecurityPolicyResponseHeaders.h>
#include <WebCore/CrossOriginAccessControl.h>
#include <WebCore/CrossOriginEmbedderPolicy.h>
#include <WebCore/FetchOptions.h>
#include <WebCore/NavigationIdentifier.h>
#include <WebCore/NavigationRequester.h>
#include <WebCore/ResourceLoaderIdentifier.h>
#include <WebCore/SecurityContext.h>
#include <wtf/Seconds.h>

namespace IPC {
class Decoder;
class Encoder;
}

namespace WebKit {

struct NetworkResourceLoadParameters {
    NetworkResourceLoadParameters(WebPageProxyIdentifier webPageProxyID, WebCore::PageIdentifier webPageID, WebCore::FrameIdentifier webFrameID)
        : webPageProxyID(webPageProxyID)
        , webPageID(webPageID)
        , webFrameID(webFrameID) { }

    NetworkResourceLoadParameters(
        WebPageProxyIdentifier
        , WebCore::PageIdentifier
        , WebCore::FrameIdentifier
        , RefPtr<WebCore::SecurityOrigin>&&
        , RefPtr<WebCore::SecurityOrigin>&&
        , WTF::ProcessID
        , WebCore::ResourceRequest&&
        , WebCore::ContentSniffingPolicy
        , WebCore::ContentEncodingSniffingPolicy
        , WebCore::StoredCredentialsPolicy
        , WebCore::ClientCredentialPolicy
        , bool shouldClearReferrerOnHTTPSToHTTPRedirect
        , bool needsCertificateInfo
        , bool isMainFrameNavigation
        , std::optional<NavigationActionData>&&
        , PreconnectOnly
        , std::optional<NavigatingToAppBoundDomain>
        , bool hadMainFrameMainResourcePrivateRelayed
        , bool allowPrivacyProxy
        , OptionSet<WebCore::AdvancedPrivacyProtections>
        , std::optional<WebCore::ResourceLoaderIdentifier>
        , RefPtr<WebCore::FormData>&& httpBody
        , std::optional<Vector<SandboxExtension::Handle>>&& sandboxExtensionIfHttpBody
        , std::optional<SandboxExtension::Handle>&& sandboxExtensionIflocalFile
        , Seconds maximumBufferingTime
        , WebCore::FetchOptions&&
        , std::optional<WebCore::ContentSecurityPolicyResponseHeaders>&& cspResponseHeaders
        , URL&& parentFrameURL
        , URL&& frameURL
        , WebCore::CrossOriginEmbedderPolicy parentCrossOriginEmbedderPolicy
        , WebCore::CrossOriginEmbedderPolicy
        , WebCore::HTTPHeaderMap&& originalRequestHeaders
        , bool shouldRestrictHTTPResponseAccess
        , WebCore::PreflightPolicy
        , bool shouldEnableCrossOriginResourcePolicy
        , Vector<Ref<WebCore::SecurityOrigin>>&& frameAncestorOrigins
        , bool pageHasResourceLoadClient
        , std::optional<WebCore::FrameIdentifier> parentFrameID
        , bool crossOriginAccessControlCheckEnabled
        , URL&& documentURL
        , bool isCrossOriginOpenerPolicyEnabled
        , bool isClearSiteDataHeaderEnabled
        , bool isClearSiteDataExecutionContextEnabled
        , bool isDisplayingInitialEmptyDocument
        , WebCore::SandboxFlags effectiveSandboxFlags
        , URL&& openerURL
        , WebCore::CrossOriginOpenerPolicy&& sourceCrossOriginOpenerPolicy
        , std::optional<WebCore::NavigationIdentifier> navigationID
        , std::optional<WebCore::NavigationRequester>&&
        , WebCore::ServiceWorkersMode
        , std::optional<WebCore::ServiceWorkerRegistrationIdentifier>
        , OptionSet<WebCore::HTTPHeadersToKeepFromCleaning>
        , std::optional<WebCore::FetchIdentifier> navigationPreloadIdentifier
#if ENABLE(CONTENT_EXTENSIONS)
        , URL&& mainDocumentURL
        , std::optional<UserContentControllerIdentifier>
#endif
#if ENABLE(WK_WEB_EXTENSIONS)
        , bool pageHasLoadedWebExtensions
#endif
        , bool linkPreconnectEarlyHintsEnabled
        , bool shouldRecordFrameLoadForStorageAccess
    );
    
    std::optional<Vector<SandboxExtension::Handle>> sandboxExtensionsIfHttpBody() const;
    std::optional<SandboxExtension::Handle> sandboxExtensionIflocalFile() const;

    RefPtr<WebCore::SecurityOrigin> parentOrigin() const;
    NetworkLoadParameters networkLoadParameters() const;

    WebPageProxyIdentifier webPageProxyID;
    WebCore::PageIdentifier webPageID;
    WebCore::FrameIdentifier webFrameID;
    RefPtr<WebCore::SecurityOrigin> topOrigin;
    RefPtr<WebCore::SecurityOrigin> sourceOrigin;
    WTF::ProcessID parentPID { 0 };
#if HAVE(AUDIT_TOKEN)
    std::optional<audit_token_t> networkProcessAuditToken;
#endif
    WebCore::ResourceRequest request;
    WebCore::ContentSniffingPolicy contentSniffingPolicy { WebCore::ContentSniffingPolicy::SniffContent };
    WebCore::ContentEncodingSniffingPolicy contentEncodingSniffingPolicy { WebCore::ContentEncodingSniffingPolicy::Default };
    WebCore::StoredCredentialsPolicy storedCredentialsPolicy { WebCore::StoredCredentialsPolicy::DoNotUse };
    WebCore::ClientCredentialPolicy clientCredentialPolicy { WebCore::ClientCredentialPolicy::CannotAskClientForCredentials };
    bool shouldClearReferrerOnHTTPSToHTTPRedirect { true };
    bool needsCertificateInfo { false };
    bool isMainFrameNavigation { false };
    std::optional<NavigationActionData> mainResourceNavigationDataForAnyFrame;
    Vector<RefPtr<WebCore::BlobDataFileReference>> blobFileReferences;
    PreconnectOnly shouldPreconnectOnly { PreconnectOnly::No };
    std::optional<NetworkActivityTracker> networkActivityTracker;
    std::optional<NavigatingToAppBoundDomain> isNavigatingToAppBoundDomain { NavigatingToAppBoundDomain::No };
    bool hadMainFrameMainResourcePrivateRelayed { false };
    bool allowPrivacyProxy { true };
    OptionSet<WebCore::AdvancedPrivacyProtections> advancedPrivacyProtections;

    RefPtr<WebCore::SecurityOrigin> protectedSourceOrigin() const { return sourceOrigin; }

    Markable<WebCore::ResourceLoaderIdentifier> identifier;
    Vector<RefPtr<SandboxExtension>> requestBodySandboxExtensions; // Created automatically for the sender.
    RefPtr<SandboxExtension> resourceSandboxExtension; // Created automatically for the sender.
    Seconds maximumBufferingTime;
    WebCore::FetchOptions options;
    std::optional<WebCore::ContentSecurityPolicyResponseHeaders> cspResponseHeaders;
    URL parentFrameURL;
    URL frameURL;
    WebCore::CrossOriginEmbedderPolicy parentCrossOriginEmbedderPolicy;
    WebCore::CrossOriginEmbedderPolicy crossOriginEmbedderPolicy;
    WebCore::HTTPHeaderMap originalRequestHeaders;
    bool shouldRestrictHTTPResponseAccess { false };
    WebCore::PreflightPolicy preflightPolicy { WebCore::PreflightPolicy::Consider };
    bool shouldEnableCrossOriginResourcePolicy { false };
    Vector<Ref<WebCore::SecurityOrigin>> frameAncestorOrigins;
    bool pageHasResourceLoadClient { false };
    std::optional<WebCore::FrameIdentifier> parentFrameID;
    bool crossOriginAccessControlCheckEnabled { true };
    URL documentURL;

    bool isCrossOriginOpenerPolicyEnabled { false };
    bool isClearSiteDataHeaderEnabled { false };
    bool isClearSiteDataExecutionContextEnabled { false };
    bool isDisplayingInitialEmptyDocument { false };
    WebCore::SandboxFlags effectiveSandboxFlags;
    URL openerURL;
    WebCore::CrossOriginOpenerPolicy sourceCrossOriginOpenerPolicy;
    std::optional<WebCore::NavigationIdentifier> navigationID;
    std::optional<WebCore::NavigationRequester> navigationRequester;

    WebCore::ServiceWorkersMode serviceWorkersMode { WebCore::ServiceWorkersMode::None };
    std::optional<WebCore::ServiceWorkerRegistrationIdentifier> serviceWorkerRegistrationIdentifier;
    OptionSet<WebCore::HTTPHeadersToKeepFromCleaning> httpHeadersToKeep;
    std::optional<WebCore::FetchIdentifier> navigationPreloadIdentifier;

#if ENABLE(CONTENT_EXTENSIONS)
    URL mainDocumentURL;
    std::optional<UserContentControllerIdentifier> userContentControllerIdentifier;
#endif

#if ENABLE(WK_WEB_EXTENSIONS)
    bool pageHasLoadedWebExtensions { false };
#endif

    bool linkPreconnectEarlyHintsEnabled { false };
    bool shouldRecordFrameLoadForStorageAccess { false };
};

} // namespace WebKit
