/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 6, 2023.
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

#if ENABLE(WK_WEB_EXTENSIONS)

#include "APIObject.h"
#include "WebExtensionTab.h"
#include "WebExtensionWindow.h"
#include <WebCore/FloatSize.h>
#include <WebCore/Icon.h>
#include <wtf/Forward.h>
#include <wtf/JSONValues.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

OBJC_CLASS NSError;
OBJC_CLASS WKWebExtensionAction;
OBJC_CLASS WKWebView;
OBJC_CLASS _WKWebExtensionActionWebView;
OBJC_CLASS _WKWebExtensionActionWebViewDelegate;

#if PLATFORM(IOS_FAMILY)
OBJC_CLASS UIViewController;
OBJC_CLASS _WKWebExtensionActionViewController;
#endif

#if PLATFORM(MAC)
OBJC_CLASS NSPopover;
OBJC_CLASS _WKWebExtensionActionPopover;
#endif

namespace WebKit {

class WebExtensionWindow;

class WebExtensionAction : public API::ObjectImpl<API::Object::Type::WebExtensionAction>, public CanMakeWeakPtr<WebExtensionAction> {
    WTF_MAKE_NONCOPYABLE(WebExtensionAction);

public:
    template<typename... Args>
    static Ref<WebExtensionAction> create(Args&&... args)
    {
        return adoptRef(*new WebExtensionAction(std::forward<Args>(args)...));
    }

    explicit WebExtensionAction(WebExtensionContext&);
    explicit WebExtensionAction(WebExtensionContext&, WebExtensionTab&);
    explicit WebExtensionAction(WebExtensionContext&, WebExtensionWindow&);

    enum class FallbackWhenEmpty { No, Yes };

#if PLATFORM(MAC)
    enum class Appearance : uint8_t { Default, Light, Dark, Both };
#endif

    bool operator==(const WebExtensionAction&) const;

    WebExtensionContext* extensionContext() const;
    RefPtr<WebExtensionTab> tab() const { return m_tab.value_or(nullptr).get(); }
    RefPtr<WebExtensionWindow> window() const { return m_window.value_or(nullptr).get(); }

    bool isTabAction() const { return !!m_tab; }
    bool isWindowAction() const { return !!m_window; }
    bool isDefaultAction() const { return !m_tab && !m_window; }

    void clearCustomizations();
    void clearBlockedResourceCount();

    void propertiesDidChange();

    RefPtr<WebCore::Icon> icon(WebCore::FloatSize idealSize);
    void setIcons(RefPtr<JSON::Object>);
#if ENABLE(WK_WEB_EXTENSIONS_ICON_VARIANTS)
    void setIconVariants(RefPtr<JSON::Array>);
#endif

    String label(FallbackWhenEmpty = FallbackWhenEmpty::Yes) const;
    void setLabel(String);

    String badgeText() const;
    void setBadgeText(String);

    bool hasUnreadBadgeText() const;
    void setHasUnreadBadgeText(bool);

    void incrementBlockedResourceCount(ssize_t amount);

    bool isEnabled() const;
    void setEnabled(std::optional<bool>);

    bool presentsPopup() const { return !popupPath().isEmpty(); }
    bool canProgrammaticallyPresentPopup() const;

    String popupPath() const;
    void setPopupPath(String);

    NSString *popupWebViewInspectionName();
    void setPopupWebViewInspectionName(const String&);

#if PLATFORM(IOS_FAMILY)
    UIViewController *popupViewController();
#endif

#if PLATFORM(MAC)
    NSPopover *popupPopover();

    Appearance popupPopoverAppearance() const { return m_popoverAppearance; }
    void setPopupPopoverAppearance(Appearance);
#endif

#if PLATFORM(COCOA)
    WKWebView *popupWebView();
    bool hasPopupWebView() const { return !!m_popupWebView; }
#endif

    bool presentsPopupWhenReady() const { return m_presentsPopupWhenReady; }
    bool popupPresented() const { return m_popupPresented; }

    void presentPopupWhenReady();
    void popupDidFinishDocumentLoad();
    void readyToPresentPopup();
    void popupSizeDidChange();
    void closePopup();

    NSArray *platformMenuItems() const;

#ifdef __OBJC__
    WKWebExtensionAction *wrapper() const { return (WKWebExtensionAction *)API::ObjectImpl<API::Object::Type::WebExtensionAction>::wrapper(); }
#endif

private:
    WebExtensionAction* fallbackAction() const;

    void clearIconCache();

#if PLATFORM(MAC)
    void detectPopoverColorScheme();
#endif

    WeakPtr<WebExtensionContext> m_extensionContext;
    std::optional<WeakPtr<WebExtensionTab>> m_tab;
    std::optional<WeakPtr<WebExtensionWindow>> m_window;

#if PLATFORM(IOS_FAMILY)
    RetainPtr<_WKWebExtensionActionViewController> m_popupViewController;
#endif

#if PLATFORM(MAC)
    RetainPtr<_WKWebExtensionActionPopover> m_popupPopover;
    Appearance m_popoverAppearance { Appearance::Default };
#endif

#if PLATFORM(COCOA)
    RetainPtr<_WKWebExtensionActionWebView> m_popupWebView;
    RetainPtr<_WKWebExtensionActionWebViewDelegate> m_popupWebViewDelegate;
#endif
    String m_customPopupPath;
    String m_popupWebViewInspectionName;

    RefPtr<WebCore::Icon> m_cachedIcon;
    Vector<double> m_cachedIconScales;
    WebCore::FloatSize m_cachedIconIdealSize;

    RefPtr<JSON::Object> m_customIcons;
#if ENABLE(WK_WEB_EXTENSIONS_ICON_VARIANTS)
    RefPtr<JSON::Array> m_customIconVariants;
#endif
    String m_customLabel;
    String m_customBadgeText;
    ssize_t m_blockedResourceCount { 0 };
    std::optional<bool> m_customEnabled;
    std::optional<bool> m_hasUnreadBadgeText;
    bool m_presentsPopupWhenReady : 1 { false };
    bool m_popupPresented : 1 { false };
    bool m_updatePending : 1 { false };
};

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
