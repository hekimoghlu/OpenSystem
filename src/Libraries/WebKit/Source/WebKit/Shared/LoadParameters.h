/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 20, 2025.
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

#include "FrameInfoData.h"
#include "NetworkResourceLoadIdentifier.h"
#include "PolicyDecision.h"
#include "SandboxExtension.h"
#include "UserData.h"
#include "WebsitePoliciesData.h"
#include <WebCore/AdvancedPrivacyProtections.h>
#include <WebCore/FrameLoaderTypes.h>
#include <WebCore/NavigationIdentifier.h>
#include <WebCore/OwnerPermissionsPolicyData.h>
#include <WebCore/PublicSuffixStore.h>
#include <WebCore/ResourceRequest.h>
#include <WebCore/ShouldTreatAsContinuingLoad.h>
#include <WebCore/SubstituteData.h>

OBJC_CLASS NSDictionary;

namespace WebCore {
class SharedBuffer;
enum class SandboxFlag : uint16_t;
using SandboxFlags = OptionSet<SandboxFlag>;
}

namespace WebKit {

struct LoadParameters {
    WebCore::PublicSuffix publicSuffix;

    std::optional<WebCore::NavigationIdentifier> navigationID;
    std::optional<WebCore::FrameIdentifier> frameIdentifier;

    WebCore::ResourceRequest request;
    SandboxExtension::Handle sandboxExtensionHandle;

    RefPtr<WebCore::SharedBuffer> data;
    String MIMEType;
    String encodingName;

    String baseURLString;
    String unreachableURLString;
    String provisionalLoadErrorURLString;

    std::optional<WebsitePoliciesData> websitePolicies;
    std::optional<FrameInfoData> originatingFrame;

    WebCore::ShouldOpenExternalURLsPolicy shouldOpenExternalURLsPolicy { WebCore::ShouldOpenExternalURLsPolicy::ShouldNotAllow };
    WebCore::ShouldTreatAsContinuingLoad shouldTreatAsContinuingLoad { WebCore::ShouldTreatAsContinuingLoad::No };
    UserData userData;
    WebCore::LockHistory lockHistory { WebCore::LockHistory::No };
    WebCore::LockBackForwardList lockBackForwardList { WebCore::LockBackForwardList::No };
    WebCore::SubstituteData::SessionHistoryVisibility sessionHistoryVisibility { WebCore::SubstituteData::SessionHistoryVisibility::Visible };
    String clientRedirectSourceForHistory;
    WebCore::SandboxFlags effectiveSandboxFlags;
    std::optional<WebCore::OwnerPermissionsPolicyData> ownerPermissionsPolicy;
    std::optional<NavigatingToAppBoundDomain> isNavigatingToAppBoundDomain;
    std::optional<NetworkResourceLoadIdentifier> existingNetworkResourceLoadIdentifierToResume;
    bool isServiceWorkerLoad { false };

#if PLATFORM(COCOA)
    std::optional<double> dataDetectionReferenceDate;
#endif
    bool isRequestFromClientOrUserInput { false };
    bool isPerformingHTTPFallback { false };

    std::optional<OptionSet<WebCore::AdvancedPrivacyProtections>> advancedPrivacyProtections;
};

} // namespace WebKit
