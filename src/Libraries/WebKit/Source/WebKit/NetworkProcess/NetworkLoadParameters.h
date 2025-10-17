/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 16, 2022.
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

#include "NavigationActionData.h"
#include "NetworkActivityTracker.h"
#include "PolicyDecision.h"
#include "WebPageProxyIdentifier.h"
#include <WebCore/BlobDataFileReference.h>
#include <WebCore/FrameIdentifier.h>
#include <WebCore/PageIdentifier.h>
#include <WebCore/ResourceLoaderOptions.h>
#include <WebCore/ResourceRequest.h>
#include <WebCore/SecurityOrigin.h>
#include <wtf/ProcessID.h>

namespace WebKit {

enum class PreconnectOnly : bool { No, Yes };

struct NetworkLoadParameters {
    Markable<WebPageProxyIdentifier> webPageProxyID;
    Markable<WebCore::PageIdentifier> webPageID;
    Markable<WebCore::FrameIdentifier> webFrameID;
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
};

} // namespace WebKit
