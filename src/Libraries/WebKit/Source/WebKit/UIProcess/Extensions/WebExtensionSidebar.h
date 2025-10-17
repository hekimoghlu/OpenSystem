/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 26, 2025.
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

#if PLATFORM(IOS_FAMILY)
OBJC_CLASS UIViewController;
using SidebarViewControllerType = UIViewController;
#endif

#if PLATFORM(MAC)
OBJC_CLASS NSViewController;
using SidebarViewControllerType = NSViewController;
#endif

#if ENABLE(WK_WEB_EXTENSIONS_SIDEBAR)

#include "APIObject.h"
#include <wtf/Forward.h>
#include <wtf/WeakHashSet.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

OBJC_CLASS WKWebView;
OBJC_CLASS _WKWebExtensionSidebar;
OBJC_CLASS _WKWebExtensionSidebarWebViewDelegate;
OBJC_CLASS _WKWebExtensionSidebarViewController;

namespace WebKit {

class WebExtensionContext;
class WebExtensionTab;
class WebExtensionWindow;

class WebExtensionSidebar : public API::ObjectImpl<API::Object::Type::WebExtensionSidebar>, public CanMakeWeakPtr<WebExtensionSidebar> {
    WTF_MAKE_NONCOPYABLE(WebExtensionSidebar);

public:
    enum class IsDefault { No, Yes };
    enum class ShouldReloadWebView { No, Yes };

    template<typename... Args>
    static Ref<WebExtensionSidebar> create(Args&&... args)
    {
        return adoptRef(*new WebExtensionSidebar(std::forward<Args>(args)...));
    }

    explicit WebExtensionSidebar(WebExtensionContext&, IsDefault = IsDefault::No);
    explicit WebExtensionSidebar(WebExtensionContext&, WebExtensionTab&);
    explicit WebExtensionSidebar(WebExtensionContext&, WebExtensionWindow&);

    bool operator==(const WebExtensionSidebar&) const;

    std::optional<Ref<WebExtensionContext>> extensionContext() const;
    const std::optional<Ref<WebExtensionTab>> tab() const;
    const std::optional<Ref<WebExtensionWindow>> window() const;
    std::optional<Ref<WebExtensionSidebar>> parent() const;

    void propertiesDidChange();

    /// `icon()` will return the overridden icon of this sidebar, or the icon of the first parent sidebar in which the icon is set
    RefPtr<WebCore::Icon> icon(WebCore::FloatSize);
    void setIconsDictionary(RefPtr<JSON::Object>);

    /// `title()` will return the overridden title of this sidebar, or the title of the first parent sidebar in which the title is set
    String title() const;
    void setTitle(std::optional<String>);

    bool isEnabled() const;
    void setEnabled(bool);

    bool isOpen() const { return m_isOpen; }
    bool opensSidebar() { return !sidebarPath().isEmpty(); };

    /// `sidebarPath()` will return the overriden path of this sidebar, or the path of the first parent sidebar in which the path is set
    String sidebarPath() const;
    void setSidebarPath(std::optional<String>);

    /// Should be called when a user action will open the sidebar
    void willOpenSidebar();
    void willCloseSidebar();

    /// Should be called when the sidebar will be displayed, regardless of whether this stems from a user action.
    void sidebarWillAppear();
    void sidebarWillDisappear();

    void addChild(WebExtensionSidebar const& child);
    void removeChild(WebExtensionSidebar const& child);

    void didReceiveUserInteraction();

    RetainPtr<SidebarViewControllerType> viewController();

    WKWebView *webView();

#ifdef __OBJC__
    _WKWebExtensionSidebar *wrapper() const { return (_WKWebExtensionSidebar *)API::ObjectImpl<API::Object::Type::WebExtensionSidebar>::wrapper(); }
#endif

private:
    explicit WebExtensionSidebar(WebExtensionContext&, std::optional<Ref<WebExtensionTab>>, std::optional<Ref<WebExtensionWindow>>, IsDefault);
    bool isDefaultSidebar() const { return m_isDefault == IsDefault::Yes; };
    bool isParentSidebar() const { return isDefaultSidebar() || m_window.has_value(); };

    void parentPropertiesWereUpdated(ShouldReloadWebView);
    void notifyChildrenOfPropertyUpdate(ShouldReloadWebView);
    void notifyDelegateOfPropertyUpdate();

    void reloadWebView();

    std::optional<RefPtr<JSON::Object>> m_iconsOverride;
    std::optional<String> m_titleOverride;
    std::optional<String> m_sidebarPathOverride;
    std::optional<bool> m_isEnabled;

    WeakPtr<WebExtensionContext> m_extensionContext;
    const std::optional<WeakPtr<WebExtensionTab>> m_tab;
    const std::optional<WeakPtr<WebExtensionWindow>> m_window;

    bool m_isOpen { false };
    const IsDefault m_isDefault { IsDefault::No };

    RetainPtr<WKWebView> m_webView;
    RetainPtr<_WKWebExtensionSidebarWebViewDelegate> m_webViewDelegate;
    RetainPtr<_WKWebExtensionSidebarViewController> m_viewController;

    WeakHashSet<WebExtensionSidebar> m_children;
};

}

#endif // ENABLE(WK_WEB_EXTENSIONS_SIDEBAR)
