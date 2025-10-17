/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 21, 2022.
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
#include "WebsitePoliciesData.h"

#include "ArgumentCoders.h"
#include "WebProcess.h"
#include <WebCore/FrameDestructionObserverInlines.h>
#include <WebCore/LocalFrame.h>
#include <WebCore/Page.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebsitePoliciesData);

void WebsitePoliciesData::applyToDocumentLoader(WebsitePoliciesData&& websitePolicies, WebCore::DocumentLoader& documentLoader)
{
    documentLoader.setCustomHeaderFields(WTFMove(websitePolicies.customHeaderFields));
    documentLoader.setCustomUserAgent(websitePolicies.customUserAgent);
    documentLoader.setCustomUserAgentAsSiteSpecificQuirks(websitePolicies.customUserAgentAsSiteSpecificQuirks);
    documentLoader.setCustomNavigatorPlatform(websitePolicies.customNavigatorPlatform);
    documentLoader.setAllowPrivacyProxy(websitePolicies.allowPrivacyProxy);

#if ENABLE(DEVICE_ORIENTATION)
    documentLoader.setDeviceOrientationAndMotionAccessState(websitePolicies.deviceOrientationAndMotionAccessState);
#endif

    // Only disable content blockers if it hasn't already been disabled by reloading without content blockers.
    auto& [defaultEnablement, exceptions] = documentLoader.contentExtensionEnablement();
    if (defaultEnablement == WebCore::ContentExtensionDefaultEnablement::Enabled && exceptions.isEmpty())
        documentLoader.setContentExtensionEnablement(WTFMove(websitePolicies.contentExtensionEnablement));

    documentLoader.setActiveContentRuleListActionPatterns(websitePolicies.activeContentRuleListActionPatterns);
    documentLoader.setVisibilityAdjustmentSelectors(WTFMove(websitePolicies.visibilityAdjustmentSelectors));

    OptionSet<WebCore::AutoplayQuirk> quirks;
    const auto& allowedQuirks = websitePolicies.allowedAutoplayQuirks;
    
    if (allowedQuirks.contains(WebsiteAutoplayQuirk::InheritedUserGestures))
        quirks.add(WebCore::AutoplayQuirk::InheritedUserGestures);
    
    if (allowedQuirks.contains(WebsiteAutoplayQuirk::SynthesizedPauseEvents))
        quirks.add(WebCore::AutoplayQuirk::SynthesizedPauseEvents);
    
    if (allowedQuirks.contains(WebsiteAutoplayQuirk::ArbitraryUserGestures))
        quirks.add(WebCore::AutoplayQuirk::ArbitraryUserGestures);

    if (allowedQuirks.contains(WebsiteAutoplayQuirk::PerDocumentAutoplayBehavior))
        quirks.add(WebCore::AutoplayQuirk::PerDocumentAutoplayBehavior);

    documentLoader.setAllowedAutoplayQuirks(quirks);

    switch (websitePolicies.autoplayPolicy) {
    case WebsiteAutoplayPolicy::Default:
        documentLoader.setAutoplayPolicy(WebCore::AutoplayPolicy::Default);
        break;
    case WebsiteAutoplayPolicy::Allow:
        documentLoader.setAutoplayPolicy(WebCore::AutoplayPolicy::Allow);
        break;
    case WebsiteAutoplayPolicy::AllowWithoutSound:
        documentLoader.setAutoplayPolicy(WebCore::AutoplayPolicy::AllowWithoutSound);
        break;
    case WebsiteAutoplayPolicy::Deny:
        documentLoader.setAutoplayPolicy(WebCore::AutoplayPolicy::Deny);
        break;
    }

    switch (websitePolicies.popUpPolicy) {
    case WebsitePopUpPolicy::Default:
        documentLoader.setPopUpPolicy(WebCore::PopUpPolicy::Default);
        break;
    case WebsitePopUpPolicy::Allow:
        documentLoader.setPopUpPolicy(WebCore::PopUpPolicy::Allow);
        break;
    case WebsitePopUpPolicy::Block:
        documentLoader.setPopUpPolicy(WebCore::PopUpPolicy::Block);
        break;
    }

    switch (websitePolicies.metaViewportPolicy) {
    case WebsiteMetaViewportPolicy::Default:
        documentLoader.setMetaViewportPolicy(WebCore::MetaViewportPolicy::Default);
        break;
    case WebsiteMetaViewportPolicy::Respect:
        documentLoader.setMetaViewportPolicy(WebCore::MetaViewportPolicy::Respect);
        break;
    case WebsiteMetaViewportPolicy::Ignore:
        documentLoader.setMetaViewportPolicy(WebCore::MetaViewportPolicy::Ignore);
        break;
    }

    switch (websitePolicies.mediaSourcePolicy) {
    case WebsiteMediaSourcePolicy::Default:
        documentLoader.setMediaSourcePolicy(WebCore::MediaSourcePolicy::Default);
        break;
    case WebsiteMediaSourcePolicy::Disable:
        documentLoader.setMediaSourcePolicy(WebCore::MediaSourcePolicy::Disable);
        break;
    case WebsiteMediaSourcePolicy::Enable:
        documentLoader.setMediaSourcePolicy(WebCore::MediaSourcePolicy::Enable);
        break;
    }

    switch (websitePolicies.simulatedMouseEventsDispatchPolicy) {
    case WebsiteSimulatedMouseEventsDispatchPolicy::Default:
        documentLoader.setSimulatedMouseEventsDispatchPolicy(WebCore::SimulatedMouseEventsDispatchPolicy::Default);
        break;
    case WebsiteSimulatedMouseEventsDispatchPolicy::Allow:
        documentLoader.setSimulatedMouseEventsDispatchPolicy(WebCore::SimulatedMouseEventsDispatchPolicy::Allow);
        break;
    case WebsiteSimulatedMouseEventsDispatchPolicy::Deny:
        documentLoader.setSimulatedMouseEventsDispatchPolicy(WebCore::SimulatedMouseEventsDispatchPolicy::Deny);
        break;
    }

    switch (websitePolicies.legacyOverflowScrollingTouchPolicy) {
    case WebsiteLegacyOverflowScrollingTouchPolicy::Default:
        documentLoader.setLegacyOverflowScrollingTouchPolicy(WebCore::LegacyOverflowScrollingTouchPolicy::Default);
        break;
    case WebsiteLegacyOverflowScrollingTouchPolicy::Disable:
        documentLoader.setLegacyOverflowScrollingTouchPolicy(WebCore::LegacyOverflowScrollingTouchPolicy::Disable);
        break;
    case WebsiteLegacyOverflowScrollingTouchPolicy::Enable:
        documentLoader.setLegacyOverflowScrollingTouchPolicy(WebCore::LegacyOverflowScrollingTouchPolicy::Enable);
        break;
    }

    switch (websitePolicies.mouseEventPolicy) {
    case WebCore::MouseEventPolicy::Default:
        documentLoader.setMouseEventPolicy(WebCore::MouseEventPolicy::Default);
        break;
#if ENABLE(IOS_TOUCH_EVENTS)
    case WebCore::MouseEventPolicy::SynthesizeTouchEvents:
        documentLoader.setMouseEventPolicy(WebCore::MouseEventPolicy::SynthesizeTouchEvents);
        break;
#endif
    }

    documentLoader.setModalContainerObservationPolicy(websitePolicies.modalContainerObservationPolicy);
    documentLoader.setColorSchemePreference(websitePolicies.colorSchemePreference);
    documentLoader.setAdvancedPrivacyProtections(websitePolicies.advancedPrivacyProtections);
    if (!documentLoader.originatorAdvancedPrivacyProtections())
        documentLoader.setOriginatorAdvancedPrivacyProtections(websitePolicies.advancedPrivacyProtections);
    documentLoader.setIdempotentModeAutosizingOnlyHonorsPercentages(websitePolicies.idempotentModeAutosizingOnlyHonorsPercentages);
    documentLoader.setHTTPSByDefaultMode(websitePolicies.httpsByDefaultMode);

    switch (websitePolicies.pushAndNotificationsEnabledPolicy) {
    case WebsitePushAndNotificationsEnabledPolicy::UseGlobalPolicy:
        documentLoader.setPushAndNotificationsEnabledPolicy(WebCore::PushAndNotificationsEnabledPolicy::UseGlobalPolicy);
        break;
    case WebsitePushAndNotificationsEnabledPolicy::No:
        documentLoader.setPushAndNotificationsEnabledPolicy(WebCore::PushAndNotificationsEnabledPolicy::No);
        break;
    case WebsitePushAndNotificationsEnabledPolicy::Yes:
        documentLoader.setPushAndNotificationsEnabledPolicy(WebCore::PushAndNotificationsEnabledPolicy::Yes);
        break;
    }

    switch (websitePolicies.inlineMediaPlaybackPolicy) {
    case WebsiteInlineMediaPlaybackPolicy::Default:
        documentLoader.setInlineMediaPlaybackPolicy(WebCore::InlineMediaPlaybackPolicy::Default);
        break;
    case WebsiteInlineMediaPlaybackPolicy::RequiresPlaysInlineAttribute:
        documentLoader.setInlineMediaPlaybackPolicy(WebCore::InlineMediaPlaybackPolicy::RequiresPlaysInlineAttribute);
        break;
    case WebsiteInlineMediaPlaybackPolicy::DoesNotRequirePlaysInlineAttribute:
        documentLoader.setInlineMediaPlaybackPolicy(WebCore::InlineMediaPlaybackPolicy::DoesNotRequirePlaysInlineAttribute);
        break;
    }

    RefPtr frame = documentLoader.frame();
    if (!frame)
        return;

    if (!frame->isMainFrame())
        return;

#if ENABLE(TOUCH_EVENTS)
    if (auto overrideValue = websitePolicies.overrideTouchEventDOMAttributesEnabled)
        frame->settings().setTouchEventDOMAttributesEnabled(*overrideValue);
#endif

    documentLoader.applyPoliciesToSettings();
}

} // namespace WebKit
