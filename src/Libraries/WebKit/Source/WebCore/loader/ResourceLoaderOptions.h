/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 14, 2025.
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

#include "ContentSecurityPolicyResponseHeaders.h"
#include "CrossOriginAccessControl.h"
#include "CrossOriginEmbedderPolicy.h"
#include "FetchIdentifier.h"
#include "FetchOptions.h"
#include "HTTPHeaderNames.h"
#include "RequestPriority.h"
#include "ServiceWorkerTypes.h"
#include "StoredCredentialsPolicy.h"
#include <wtf/HashSet.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

enum class SendCallbackPolicy : uint8_t {
    SendCallbacks,
    DoNotSendCallbacks
};
static constexpr unsigned bitWidthOfSendCallbackPolicy = 1;

// FIXME: These options are named poorly. We only implement force disabling content sniffing, not enabling it,
// and even that only on some platforms.
enum class ContentSniffingPolicy : bool {
    SniffContent,
    DoNotSniffContent
};
static constexpr unsigned bitWidthOfContentSniffingPolicy = 1;

enum class DataBufferingPolicy : uint8_t {
    BufferData,
    DoNotBufferData
};
static constexpr unsigned bitWidthOfDataBufferingPolicy = 1;

enum class SecurityCheckPolicy : uint8_t {
    SkipSecurityCheck,
    DoSecurityCheck
};
static constexpr unsigned bitWidthOfSecurityCheckPolicy = 1;

enum class CertificateInfoPolicy : uint8_t {
    IncludeCertificateInfo,
    DoNotIncludeCertificateInfo
};
static constexpr unsigned bitWidthOfCertificateInfoPolicy = 1;

enum class ContentSecurityPolicyImposition : uint8_t {
    SkipPolicyCheck,
    DoPolicyCheck
};
static constexpr unsigned bitWidthOfContentSecurityPolicyImposition = 1;

enum class DefersLoadingPolicy : uint8_t {
    AllowDefersLoading,
    DisallowDefersLoading
};
static constexpr unsigned bitWidthOfDefersLoadingPolicy = 1;

enum class CachingPolicy : uint8_t {
    AllowCaching,
    DisallowCaching
};
static constexpr unsigned bitWidthOfCachingPolicy = 1;

enum class ClientCredentialPolicy : bool {
    CannotAskClientForCredentials,
    MayAskClientForCredentials
};
static constexpr unsigned bitWidthOfClientCredentialPolicy = 1;

enum class SameOriginDataURLFlag : uint8_t {
    Set,
    Unset
};
static constexpr unsigned bitWidthOfSameOriginDataURLFlag = 1;

enum class InitiatorContext : uint8_t {
    Document,
    Worker,
};
static constexpr unsigned bitWidthOfInitiatorContext = 1;

// https://fetch.spec.whatwg.org/#concept-request-initiator
enum class Initiator : uint8_t {
    EmptyString,
    Download,
    Imageset,
    Manifest,
    Prefetch,
    Prerender,
    Xslt
};
static constexpr unsigned bitWidthOfInitiator = 3;

enum class ServiceWorkersMode : uint8_t {
    All,
    None,
    Only // An error will happen if service worker is not handling the fetch. Used to bypass preflight safely.
};
static constexpr unsigned bitWidthOfServiceWorkersMode = 2;

enum class ApplicationCacheMode : uint8_t {
    Use,
    Bypass
};
static constexpr unsigned bitWidthOfApplicationCacheMode = 1;

enum class ContentEncodingSniffingPolicy : bool {
    Default,
    Disable
};
static constexpr unsigned bitWidthOfContentEncodingSniffingPolicy = 1;

enum class PreflightPolicy : uint8_t {
    Consider,
    Force,
    Prevent
};
static constexpr unsigned bitWidthOfPreflightPolicy = 2;

enum class ShouldEnableContentExtensionsCheck : bool { No, Yes };
static constexpr unsigned bitWidthOfShouldEnableContentExtensionsCheck = 1;

enum class LoadedFromOpaqueSource : bool { No, Yes };
static constexpr unsigned bitWidthOfLoadedFromOpaqueSource = 1;

enum class LoadedFromPluginElement : bool { No, Yes };
static constexpr unsigned bitWidthOfLoadedFromPluginElement = 1;

enum class LoadedFromFetch : bool { No, Yes };
static constexpr unsigned bitWidthOfLoadedFromFetch = 1;

struct ResourceLoaderOptions : public FetchOptions {
    ResourceLoaderOptions()
        : ResourceLoaderOptions(FetchOptions())
    {
    }

    ResourceLoaderOptions(FetchOptions options)
        : FetchOptions { WTFMove(options) }
        , sendLoadCallbacks(SendCallbackPolicy::DoNotSendCallbacks)
        , sniffContent(ContentSniffingPolicy::DoNotSniffContent)
        , contentEncodingSniffingPolicy(ContentEncodingSniffingPolicy::Default)
        , dataBufferingPolicy(DataBufferingPolicy::BufferData)
        , storedCredentialsPolicy(StoredCredentialsPolicy::DoNotUse)
        , securityCheck(SecurityCheckPolicy::DoSecurityCheck)
        , certificateInfoPolicy(CertificateInfoPolicy::DoNotIncludeCertificateInfo)
        , contentSecurityPolicyImposition(ContentSecurityPolicyImposition::DoPolicyCheck)
        , defersLoadingPolicy(DefersLoadingPolicy::AllowDefersLoading)
        , cachingPolicy(CachingPolicy::AllowCaching)
        , sameOriginDataURLFlag(SameOriginDataURLFlag::Unset)
        , initiatorContext(InitiatorContext::Document)
        , initiator(Initiator::EmptyString)
        , serviceWorkersMode(ServiceWorkersMode::All)
        , applicationCacheMode(ApplicationCacheMode::Use)
        , clientCredentialPolicy(ClientCredentialPolicy::CannotAskClientForCredentials)
        , preflightPolicy(PreflightPolicy::Consider)
        , loadedFromOpaqueSource(LoadedFromOpaqueSource::No)
        , loadedFromPluginElement(LoadedFromPluginElement::No)
        , loadedFromFetch(LoadedFromFetch::No)
        , fetchPriority(RequestPriority::Auto)
        , shouldEnableContentExtensionsCheck(ShouldEnableContentExtensionsCheck::Yes)
    { }

    ResourceLoaderOptions(SendCallbackPolicy sendLoadCallbacks, ContentSniffingPolicy sniffContent, DataBufferingPolicy dataBufferingPolicy, StoredCredentialsPolicy storedCredentialsPolicy, ClientCredentialPolicy credentialPolicy, FetchOptions::Credentials credentials, SecurityCheckPolicy securityCheck, FetchOptions::Mode mode, CertificateInfoPolicy certificateInfoPolicy, ContentSecurityPolicyImposition contentSecurityPolicyImposition, DefersLoadingPolicy defersLoadingPolicy, CachingPolicy cachingPolicy)
        : sendLoadCallbacks(sendLoadCallbacks)
        , sniffContent(sniffContent)
        , contentEncodingSniffingPolicy(ContentEncodingSniffingPolicy::Default)
        , dataBufferingPolicy(dataBufferingPolicy)
        , storedCredentialsPolicy(storedCredentialsPolicy)
        , securityCheck(securityCheck)
        , certificateInfoPolicy(certificateInfoPolicy)
        , contentSecurityPolicyImposition(contentSecurityPolicyImposition)
        , defersLoadingPolicy(defersLoadingPolicy)
        , cachingPolicy(cachingPolicy)
        , sameOriginDataURLFlag(SameOriginDataURLFlag::Unset)
        , initiatorContext(InitiatorContext::Document)
        , initiator(Initiator::EmptyString)
        , serviceWorkersMode(ServiceWorkersMode::All)
        , applicationCacheMode(ApplicationCacheMode::Use)
        , clientCredentialPolicy(credentialPolicy)
        , preflightPolicy(PreflightPolicy::Consider)
        , loadedFromOpaqueSource(LoadedFromOpaqueSource::No)
        , loadedFromPluginElement(LoadedFromPluginElement::No)
        , loadedFromFetch(LoadedFromFetch::No)
        , fetchPriority(RequestPriority::Auto)
        , shouldEnableContentExtensionsCheck(ShouldEnableContentExtensionsCheck::Yes)
    {
        this->credentials = credentials;
        this->mode = mode;
    }

    Markable<ServiceWorkerRegistrationIdentifier, ServiceWorkerRegistrationIdentifier::MarkableTraits> serviceWorkerRegistrationIdentifier;
    Markable<ContentSecurityPolicyResponseHeaders, ContentSecurityPolicyResponseHeaders::MarkableTraits> cspResponseHeaders;
    std::optional<CrossOriginEmbedderPolicy> crossOriginEmbedderPolicy;

    uint8_t maxRedirectCount { 20 };
    OptionSet<HTTPHeadersToKeepFromCleaning> httpHeadersToKeep;

    SendCallbackPolicy sendLoadCallbacks : bitWidthOfSendCallbackPolicy;
    ContentSniffingPolicy sniffContent : bitWidthOfContentSniffingPolicy;
    ContentEncodingSniffingPolicy contentEncodingSniffingPolicy : bitWidthOfContentEncodingSniffingPolicy;
    DataBufferingPolicy dataBufferingPolicy : bitWidthOfDataBufferingPolicy;
    StoredCredentialsPolicy storedCredentialsPolicy : bitWidthOfStoredCredentialsPolicy;
    SecurityCheckPolicy securityCheck : bitWidthOfSecurityCheckPolicy;
    CertificateInfoPolicy certificateInfoPolicy : bitWidthOfCertificateInfoPolicy;
    ContentSecurityPolicyImposition contentSecurityPolicyImposition : bitWidthOfContentSecurityPolicyImposition;
    DefersLoadingPolicy defersLoadingPolicy : bitWidthOfDefersLoadingPolicy;
    CachingPolicy cachingPolicy : bitWidthOfCachingPolicy;
    SameOriginDataURLFlag sameOriginDataURLFlag : bitWidthOfSameOriginDataURLFlag;
    InitiatorContext initiatorContext : bitWidthOfInitiatorContext;
    Initiator initiator : bitWidthOfInitiator;
    ServiceWorkersMode serviceWorkersMode : bitWidthOfServiceWorkersMode;
    ApplicationCacheMode applicationCacheMode : bitWidthOfApplicationCacheMode;
    ClientCredentialPolicy clientCredentialPolicy : bitWidthOfClientCredentialPolicy;
    PreflightPolicy preflightPolicy : bitWidthOfPreflightPolicy;
    LoadedFromOpaqueSource loadedFromOpaqueSource : bitWidthOfLoadedFromOpaqueSource;
    LoadedFromPluginElement loadedFromPluginElement : bitWidthOfLoadedFromPluginElement;
    LoadedFromFetch loadedFromFetch : bitWidthOfLoadedFromFetch;
    RequestPriority fetchPriority : bitWidthOfRequestPriority;
    ShouldEnableContentExtensionsCheck shouldEnableContentExtensionsCheck : bitWidthOfShouldEnableContentExtensionsCheck;

    Markable<FetchIdentifier> navigationPreloadIdentifier;
    String nonce;
};

} // namespace WebCore
