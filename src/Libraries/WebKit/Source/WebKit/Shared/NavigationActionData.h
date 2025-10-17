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
#pragma once

#include "FrameInfoData.h"
#include "WebHitTestResultData.h"
#include "WebMouseEvent.h"
#include "WebPageProxyIdentifier.h"
#include <WebCore/AdvancedPrivacyProtections.h>
#include <WebCore/BackForwardItemIdentifier.h>
#include <WebCore/FloatPoint.h>
#include <WebCore/FrameLoaderTypes.h>
#include <WebCore/NavigationIdentifier.h>
#include <WebCore/OwnerPermissionsPolicyData.h>
#include <WebCore/PrivateClickMeasurement.h>
#include <WebCore/ResourceRequest.h>
#include <WebCore/ResourceResponse.h>
#include <WebCore/SandboxFlags.h>
#include <WebCore/SecurityOriginData.h>
#include <WebCore/UserGestureTokenIdentifier.h>

namespace WebKit {

struct NavigationActionData {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;

    WebCore::NavigationType navigationType { WebCore::NavigationType::Other };
    OptionSet<WebEventModifier> modifiers;
    WebMouseEventButton mouseButton { WebMouseEventButton::None };
    WebMouseEventSyntheticClickType syntheticClickType { WebMouseEventSyntheticClickType::NoTap };
    std::optional<WebCore::UserGestureTokenIdentifier> userGestureTokenIdentifier;
    std::optional<WTF::UUID> userGestureAuthorizationToken;
    bool canHandleRequest { false };
    WebCore::ShouldOpenExternalURLsPolicy shouldOpenExternalURLsPolicy { WebCore::ShouldOpenExternalURLsPolicy::ShouldNotAllow };
    WTF::String downloadAttribute;
    WebCore::FloatPoint clickLocationInRootViewCoordinates;
    WebCore::ResourceResponse redirectResponse;
    bool isRequestFromClientOrUserInput { false };
    bool treatAsSameOriginNavigation { false };
    bool hasOpenedFrames { false };
    bool openedByDOMWithOpener { false };
    bool hasOpener { false };
    bool isPerformingHTTPFallback { false };
    String openedMainFrameName;
    WebCore::SecurityOriginData requesterOrigin;
    WebCore::SecurityOriginData requesterTopOrigin;
    std::optional<WebCore::BackForwardItemIdentifier> targetBackForwardItemIdentifier;
    std::optional<WebCore::BackForwardItemIdentifier> sourceBackForwardItemIdentifier;
    WebCore::LockHistory lockHistory { WebCore::LockHistory::No };
    WebCore::LockBackForwardList lockBackForwardList { WebCore::LockBackForwardList::No };
    WTF::String clientRedirectSourceForHistory;
    WebCore::SandboxFlags effectiveSandboxFlags;
    std::optional<WebCore::OwnerPermissionsPolicyData> ownerPermissionsPolicy;
    std::optional<WebCore::PrivateClickMeasurement> privateClickMeasurement;
    OptionSet<WebCore::AdvancedPrivacyProtections> advancedPrivacyProtections;
    std::optional<OptionSet<WebCore::AdvancedPrivacyProtections>> originatorAdvancedPrivacyProtections;
#if PLATFORM(MAC) || HAVE(UIKIT_WITH_MOUSE_SUPPORT)
    std::optional<WebKit::WebHitTestResultData> webHitTestResultData;
#endif
    FrameInfoData originatingFrameInfoData;
    std::optional<WebPageProxyIdentifier> originatingPageID;
    FrameInfoData frameInfo;
    std::optional<WebCore::NavigationIdentifier> navigationID;
    WebCore::ResourceRequest originalRequest;
    WebCore::ResourceRequest request;
};

}
