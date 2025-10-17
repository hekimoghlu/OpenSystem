/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 24, 2023.
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

#include "WebExtensionError.h"
#include "WebExtensionWindowIdentifier.h"
#include "WebPageProxyIdentifier.h"
#include <wtf/Forward.h>
#include <wtf/Identified.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/TZoneMalloc.h>
#if PLATFORM(COCOA)
#include <wtf/WeakObjCPtr.h>
#endif

OBJC_PROTOCOL(WKWebExtensionWindow);

#ifdef __OBJC__
#import "WKWebExtensionWindow.h"
#endif

namespace WebKit {

class WebExtensionContext;
class WebExtensionTab;
struct WebExtensionTabQueryParameters;
struct WebExtensionWindowParameters;

enum class WebExtensionWindowTypeFilter : uint8_t {
    Normal = 1 << 0,
    Popup  = 1 << 1,
};

static constexpr OptionSet<WebExtensionWindowTypeFilter> allWebExtensionWindowTypeFilters()
{
    return {
        WebExtensionWindowTypeFilter::Normal,
        WebExtensionWindowTypeFilter::Popup
    };
}

class WebExtensionWindow : public RefCountedAndCanMakeWeakPtr<WebExtensionWindow>, public Identified<WebExtensionWindowIdentifier> {
    WTF_MAKE_NONCOPYABLE(WebExtensionWindow);
    WTF_MAKE_TZONE_ALLOCATED(WebExtensionWindow);

public:
    template<typename... Args>
    static Ref<WebExtensionWindow> create(Args&&... args)
    {
        return adoptRef(*new WebExtensionWindow(std::forward<Args>(args)...));
    }

    explicit WebExtensionWindow(const WebExtensionContext&, WKWebExtensionWindow*);

    enum class Type : uint8_t {
        Normal,
        Popup,
    };

    using TypeFilter = WebExtensionWindowTypeFilter;

    enum class State : uint8_t {
        Normal,
        Minimized,
        Maximized,
        Fullscreen,
    };

    enum class PopulateTabs : bool { No, Yes };
    enum class SkipValidation : bool { No, Yes };

    using TabVector = Vector<Ref<WebExtensionTab>>;

    WebExtensionWindowParameters parameters(PopulateTabs = PopulateTabs::No) const;
    WebExtensionWindowParameters minimalParameters() const;

    WebExtensionContext* extensionContext() const;

    bool operator==(const WebExtensionWindow&) const;

    bool matches(OptionSet<TypeFilter>) const;
    bool matches(const WebExtensionTabQueryParameters&, std::optional<WebPageProxyIdentifier> = std::nullopt) const;

    bool extensionHasAccess() const;

    TabVector tabs(SkipValidation = SkipValidation::No) const;
    RefPtr<WebExtensionTab> activeTab(SkipValidation = SkipValidation::No) const;

    Type type() const;

    State state() const;
    void setState(State, CompletionHandler<void(Expected<void, WebExtensionError>&&)>&&);

    bool isOpen() const;
    void didOpen() { ASSERT(!m_isOpen); m_isOpen = true; }
    void didClose() { ASSERT(m_isOpen); m_isOpen = false; }

    bool isFocused() const;
    bool isFrontmost() const;
    void focus(CompletionHandler<void(Expected<void, WebExtensionError>&&)>&&);

    bool isPrivate() const;

#if PLATFORM(COCOA)
    // Returns the frame using top-down coordinates.
    CGRect normalizedFrame() const;

    // Handles the frame in the screen's native coordinate system.
    CGRect frame() const;
    void setFrame(CGRect, CompletionHandler<void(Expected<void, WebExtensionError>&&)>&&);
#endif

#if PLATFORM(MAC)
    CGRect screenFrame() const;
#endif

    void close(CompletionHandler<void(Expected<void, WebExtensionError>&&)>&&);

#ifdef __OBJC__
    WKWebExtensionWindow *delegate() const { return m_delegate.getAutoreleased(); }

    bool isValid() const { return m_extensionContext && m_delegate; }
#endif

private:
    WeakPtr<WebExtensionContext> m_extensionContext;
#if PLATFORM(COCOA)
    WeakObjCPtr<WKWebExtensionWindow> m_delegate;
#endif
    bool m_isOpen : 1 { false };
    mutable bool m_private : 1 { false };
    mutable bool m_cachedPrivate : 1 { false };
    bool m_respondsToTabs : 1 { false };
    bool m_respondsToActiveTab : 1 { false };
    bool m_respondsToWindowType : 1 { false };
    bool m_respondsToWindowState : 1 { false };
    bool m_respondsToSetWindowState : 1 { false };
    bool m_respondsToIsPrivate : 1 { false };
    bool m_respondsToFrame : 1 { false };
    bool m_respondsToSetFrame : 1 { false };
    bool m_respondsToScreenFrame : 1 { false };
    bool m_respondsToFocus : 1 { false };
    bool m_respondsToClose : 1 { false };
};

#ifdef __OBJC__
WKWebExtensionWindowType toAPI(WebExtensionWindow::Type);
WKWebExtensionWindowState toAPI(WebExtensionWindow::State);
#endif

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
