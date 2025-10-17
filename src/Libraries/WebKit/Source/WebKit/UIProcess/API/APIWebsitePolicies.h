/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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

#include "APIObject.h"
#include "WebsitePoliciesData.h"
#include <wtf/WeakPtr.h>

namespace WebKit {
class LockdownModeObserver;
class WebUserContentControllerProxy;
class WebsiteDataStore;
struct WebsitePoliciesData;
}

namespace API {

class WebsitePolicies final : public API::ObjectImpl<API::Object::Type::WebsitePolicies>, public CanMakeWeakPtr<WebsitePolicies> {
public:
    static Ref<WebsitePolicies> create() { return adoptRef(*new WebsitePolicies); }
    WebsitePolicies();
    ~WebsitePolicies();

    Ref<WebsitePolicies> copy() const;

    WebKit::WebsitePoliciesData data();

    const WebCore::ContentExtensionEnablement& contentExtensionEnablement() const { return m_data.contentExtensionEnablement; }
    void setContentExtensionEnablement(WebCore::ContentExtensionEnablement&& enablement) { m_data.contentExtensionEnablement = WTFMove(enablement); }

    void setActiveContentRuleListActionPatterns(HashMap<WTF::String, Vector<WTF::String>>&& patterns) { m_data.activeContentRuleListActionPatterns = WTFMove(patterns); }
    const HashMap<WTF::String, Vector<WTF::String>>& activeContentRuleListActionPatterns() const { return m_data.activeContentRuleListActionPatterns; }
    
    OptionSet<WebKit::WebsiteAutoplayQuirk> allowedAutoplayQuirks() const { return m_data.allowedAutoplayQuirks; }
    void setAllowedAutoplayQuirks(OptionSet<WebKit::WebsiteAutoplayQuirk> quirks) { m_data.allowedAutoplayQuirks = quirks; }
    
    WebKit::WebsiteAutoplayPolicy autoplayPolicy() const { return m_data.autoplayPolicy; }
    void setAutoplayPolicy(WebKit::WebsiteAutoplayPolicy policy) { m_data.autoplayPolicy = policy; }

#if ENABLE(DEVICE_ORIENTATION)
    WebCore::DeviceOrientationOrMotionPermissionState deviceOrientationAndMotionAccessState() const { return m_data.deviceOrientationAndMotionAccessState; }
    void setDeviceOrientationAndMotionAccessState(WebCore::DeviceOrientationOrMotionPermissionState state) { m_data.deviceOrientationAndMotionAccessState = state; }
#endif

    const Vector<WebCore::CustomHeaderFields>& customHeaderFields() const { return m_data.customHeaderFields; }
    void setCustomHeaderFields(Vector<WebCore::CustomHeaderFields>&& fields) { m_data.customHeaderFields = WTFMove(fields); }

    WebKit::WebsitePopUpPolicy popUpPolicy() const { return m_data.popUpPolicy; }
    void setPopUpPolicy(WebKit::WebsitePopUpPolicy policy) { m_data.popUpPolicy = policy; }

    WebKit::WebsiteDataStore* websiteDataStore() const { return m_websiteDataStore.get(); }
    void setWebsiteDataStore(RefPtr<WebKit::WebsiteDataStore>&&);
    
    WebKit::WebUserContentControllerProxy* userContentController() const { return m_userContentController.get(); }
    void setUserContentController(RefPtr<WebKit::WebUserContentControllerProxy>&&);

    void setCustomUserAgent(const WTF::String& customUserAgent) { m_data.customUserAgent = customUserAgent; }
    const WTF::String& customUserAgent() const { return m_data.customUserAgent; }

    void setCustomUserAgentAsSiteSpecificQuirks(const WTF::String& customUserAgent) { m_data.customUserAgentAsSiteSpecificQuirks = customUserAgent; }
    const WTF::String& customUserAgentAsSiteSpecificQuirks() const { return m_data.customUserAgentAsSiteSpecificQuirks; }

    void setCustomNavigatorPlatform(const WTF::String& customNavigatorPlatform) { m_data.customNavigatorPlatform = customNavigatorPlatform; }
    const WTF::String& customNavigatorPlatform() const { return m_data.customNavigatorPlatform; }

    WebKit::WebContentMode preferredContentMode() const { return m_data.preferredContentMode; }
    void setPreferredContentMode(WebKit::WebContentMode mode) { m_data.preferredContentMode = mode; }

    WebKit::WebsiteMetaViewportPolicy metaViewportPolicy() const { return m_data.metaViewportPolicy; }
    void setMetaViewportPolicy(WebKit::WebsiteMetaViewportPolicy policy) { m_data.metaViewportPolicy = policy; }

    WebKit::WebsiteMediaSourcePolicy mediaSourcePolicy() const { return m_data.mediaSourcePolicy; }
    void setMediaSourcePolicy(WebKit::WebsiteMediaSourcePolicy policy) { m_data.mediaSourcePolicy = policy; }

    WebKit::WebsiteSimulatedMouseEventsDispatchPolicy simulatedMouseEventsDispatchPolicy() const { return m_data.simulatedMouseEventsDispatchPolicy; }
    void setSimulatedMouseEventsDispatchPolicy(WebKit::WebsiteSimulatedMouseEventsDispatchPolicy policy) { m_data.simulatedMouseEventsDispatchPolicy = policy; }

    WebKit::WebsiteLegacyOverflowScrollingTouchPolicy legacyOverflowScrollingTouchPolicy() const { return m_data.legacyOverflowScrollingTouchPolicy; }
    void setLegacyOverflowScrollingTouchPolicy(WebKit::WebsiteLegacyOverflowScrollingTouchPolicy policy) { m_data.legacyOverflowScrollingTouchPolicy = policy; }

    bool allowSiteSpecificQuirksToOverrideContentMode() const { return m_data.allowSiteSpecificQuirksToOverrideContentMode; }
    void setAllowSiteSpecificQuirksToOverrideContentMode(bool value) { m_data.allowSiteSpecificQuirksToOverrideContentMode = value; }

    WTF::String applicationNameForDesktopUserAgent() const { return m_data.applicationNameForDesktopUserAgent; }
    void setApplicationNameForDesktopUserAgent(const WTF::String& applicationName) { m_data.applicationNameForDesktopUserAgent = applicationName; }

    WebCore::AllowsContentJavaScript allowsContentJavaScript() const { return m_data.allowsContentJavaScript; }
    void setAllowsContentJavaScript(WebCore::AllowsContentJavaScript allows) { m_data.allowsContentJavaScript = allows; }

    bool lockdownModeEnabled() const;
    void setLockdownModeEnabled(std::optional<bool> enabled) { m_lockdownModeEnabled = enabled; }
    bool isLockdownModeExplicitlySet() const { return !!m_lockdownModeEnabled; }

    WebCore::ColorSchemePreference colorSchemePreference() const { return m_data.colorSchemePreference; }
    void setColorSchemePreference(WebCore::ColorSchemePreference colorSchemePreference) { m_data.colorSchemePreference = colorSchemePreference; }

    WebCore::MouseEventPolicy mouseEventPolicy() const { return m_data.mouseEventPolicy; }
    void setMouseEventPolicy(WebCore::MouseEventPolicy policy) { m_data.mouseEventPolicy = policy; }

    WebCore::ModalContainerObservationPolicy modalContainerObservationPolicy() const { return m_data.modalContainerObservationPolicy; }
    void setModalContainerObservationPolicy(WebCore::ModalContainerObservationPolicy policy) { m_data.modalContainerObservationPolicy = policy; }

    OptionSet<WebCore::AdvancedPrivacyProtections> advancedPrivacyProtections() const { return m_data.advancedPrivacyProtections; }
    void setAdvancedPrivacyProtections(OptionSet<WebCore::AdvancedPrivacyProtections> policy) { m_data.advancedPrivacyProtections = policy; }

    bool idempotentModeAutosizingOnlyHonorsPercentages() const { return m_data.idempotentModeAutosizingOnlyHonorsPercentages; }
    void setIdempotentModeAutosizingOnlyHonorsPercentages(bool idempotentModeAutosizingOnlyHonorsPercentages) { m_data.idempotentModeAutosizingOnlyHonorsPercentages = idempotentModeAutosizingOnlyHonorsPercentages; }

    bool allowPrivacyProxy() const { return m_data.allowPrivacyProxy; }
    void setAllowPrivacyProxy(bool allow) { m_data.allowPrivacyProxy = allow; }

    WebCore::HTTPSByDefaultMode httpsByDefaultMode() const { return m_data.httpsByDefaultMode; }
    void setHTTPSByDefault(WebCore::HTTPSByDefaultMode mode) { m_data.httpsByDefaultMode = mode; }
    bool isUpgradeWithUserMediatedFallbackEnabled() { return advancedPrivacyProtections().contains(WebCore::AdvancedPrivacyProtections::HTTPSOnly) || httpsByDefaultMode() == WebCore::HTTPSByDefaultMode::UpgradeWithUserMediatedFallback; }
    bool isUpgradeWithAutomaticFallbackEnabled() { return advancedPrivacyProtections().contains(WebCore::AdvancedPrivacyProtections::HTTPSFirst) || httpsByDefaultMode() == WebCore::HTTPSByDefaultMode::UpgradeWithAutomaticFallback; }

    const Vector<Vector<HashSet<WTF::String>>>& visibilityAdjustmentSelectors() const { return m_data.visibilityAdjustmentSelectors; }
    void setVisibilityAdjustmentSelectors(Vector<Vector<HashSet<WTF::String>>>&& selectors) { m_data.visibilityAdjustmentSelectors = WTFMove(selectors); }

    WebKit::WebsitePushAndNotificationsEnabledPolicy pushAndNotificationsEnabledPolicy() const { return m_data.pushAndNotificationsEnabledPolicy; }
    void setPushAndNotificationsEnabledPolicy(WebKit::WebsitePushAndNotificationsEnabledPolicy policy) { m_data.pushAndNotificationsEnabledPolicy = policy; }

#if ENABLE(TOUCH_EVENTS)
    void setOverrideTouchEventDOMAttributesEnabled(bool value) { m_data.overrideTouchEventDOMAttributesEnabled = value; }
#endif

    WebKit::WebsiteInlineMediaPlaybackPolicy inlineMediaPlaybackPolicy() const { return m_data.inlineMediaPlaybackPolicy; }
    void setInlineMediaPlaybackPolicy(WebKit::WebsiteInlineMediaPlaybackPolicy policy) { m_data.inlineMediaPlaybackPolicy = policy; }

private:
    WebKit::WebsitePoliciesData m_data;
    RefPtr<WebKit::WebsiteDataStore> m_websiteDataStore;
    RefPtr<WebKit::WebUserContentControllerProxy> m_userContentController;
    std::optional<bool> m_lockdownModeEnabled;
#if PLATFORM(COCOA)
    std::unique_ptr<WebKit::LockdownModeObserver> m_lockdownModeObserver;
#endif
};

} // namespace API
