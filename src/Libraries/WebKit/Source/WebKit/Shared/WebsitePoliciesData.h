/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 17, 2025.
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

#include "WebContentMode.h"
#include "WebsiteAutoplayPolicy.h"
#include "WebsiteAutoplayQuirk.h"
#include "WebsiteInlineMediaPlaybackPolicy.h"
#include "WebsiteLegacyOverflowScrollingTouchPolicy.h"
#include "WebsiteMediaSourcePolicy.h"
#include "WebsiteMetaViewportPolicy.h"
#include "WebsitePopUpPolicy.h"
#include "WebsitePushAndNotificationsEnabledPolicy.h"
#include "WebsiteSimulatedMouseEventsDispatchPolicy.h"
#include <WebCore/AdvancedPrivacyProtections.h>
#include <WebCore/CustomHeaderFields.h>
#include <WebCore/DeviceOrientationOrMotionPermissionState.h>
#include <WebCore/DocumentLoader.h>
#include <WebCore/FrameLoaderTypes.h>
#include <wtf/HashSet.h>
#include <wtf/OptionSet.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {
class DocumentLoader;
}

namespace WebKit {

struct WebsitePoliciesData {
    WTF_MAKE_TZONE_ALLOCATED(WebsitePoliciesData);
public:
    static void applyToDocumentLoader(WebsitePoliciesData&&, WebCore::DocumentLoader&);

    HashMap<String, Vector<String>> activeContentRuleListActionPatterns;
    Vector<WebCore::CustomHeaderFields> customHeaderFields;
    Vector<WebCore::TargetedElementSelectors> visibilityAdjustmentSelectors;
    String customUserAgent;
    String customUserAgentAsSiteSpecificQuirks;
    String customNavigatorPlatform;
    String applicationNameForDesktopUserAgent;
    OptionSet<WebCore::AdvancedPrivacyProtections> advancedPrivacyProtections;
    OptionSet<WebsiteAutoplayQuirk> allowedAutoplayQuirks;
    WebCore::ContentExtensionEnablement contentExtensionEnablement { WebCore::ContentExtensionDefaultEnablement::Enabled, { } };
#if ENABLE(TOUCH_EVENTS)
    std::optional<bool> overrideTouchEventDOMAttributesEnabled;
#endif
    WebsiteAutoplayPolicy autoplayPolicy { WebsiteAutoplayPolicy::Default };
    WebsitePopUpPolicy popUpPolicy { WebsitePopUpPolicy::Default };
    WebsiteMetaViewportPolicy metaViewportPolicy { WebsiteMetaViewportPolicy::Default };
    WebsiteMediaSourcePolicy mediaSourcePolicy { WebsiteMediaSourcePolicy::Default };
    WebsiteSimulatedMouseEventsDispatchPolicy simulatedMouseEventsDispatchPolicy { WebsiteSimulatedMouseEventsDispatchPolicy::Default };
    WebsiteLegacyOverflowScrollingTouchPolicy legacyOverflowScrollingTouchPolicy { WebsiteLegacyOverflowScrollingTouchPolicy::Default };
    WebCore::AllowsContentJavaScript allowsContentJavaScript { WebCore::AllowsContentJavaScript::Yes };
    WebCore::MouseEventPolicy mouseEventPolicy { WebCore::MouseEventPolicy::Default };
    WebCore::ModalContainerObservationPolicy modalContainerObservationPolicy { WebCore::ModalContainerObservationPolicy::Disabled };
    WebCore::ColorSchemePreference colorSchemePreference { WebCore::ColorSchemePreference::NoPreference };
    WebContentMode preferredContentMode { WebContentMode::Recommended };
#if ENABLE(DEVICE_ORIENTATION)
    WebCore::DeviceOrientationOrMotionPermissionState deviceOrientationAndMotionAccessState { WebCore::DeviceOrientationOrMotionPermissionState::Prompt };
#endif
    WebCore::HTTPSByDefaultMode httpsByDefaultMode { WebCore::HTTPSByDefaultMode::Disabled };
    bool idempotentModeAutosizingOnlyHonorsPercentages { false };
    bool allowPrivacyProxy { true };
    bool allowSiteSpecificQuirksToOverrideContentMode { false };
    WebsitePushAndNotificationsEnabledPolicy pushAndNotificationsEnabledPolicy { WebsitePushAndNotificationsEnabledPolicy::UseGlobalPolicy };
    WebsiteInlineMediaPlaybackPolicy inlineMediaPlaybackPolicy { WebsiteInlineMediaPlaybackPolicy::Default };
};

} // namespace WebKit
