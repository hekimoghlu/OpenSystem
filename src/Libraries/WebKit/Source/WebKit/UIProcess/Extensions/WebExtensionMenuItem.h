/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 21, 2021.
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

#include "WebExtension.h"
#include "WebExtensionCommand.h"
#include "WebExtensionMenuItemContextType.h"
#include "WebExtensionMenuItemType.h"
#include <wtf/Forward.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/WTFString.h>

OBJC_CLASS NSArray;

#if USE(APPKIT)
OBJC_CLASS NSMenuItem;
using CocoaMenuItem = NSMenuItem;
#else
OBJC_CLASS UIMenuElement;
using CocoaMenuItem = UIMenuElement;
#endif

#if defined(__OBJC__) && USE(APPKIT)
using WebExtensionMenuItemHandlerBlock = void (^)(id);

@interface _WKWebExtensionMenuItem : NSMenuItem

- (instancetype)initWithTitle:(NSString *)title handler:(WebExtensionMenuItemHandlerBlock)block;

@property (nonatomic, copy) WebExtensionMenuItemHandlerBlock handler;

@end
#endif // USE(APPKIT)

namespace WebKit {

class WebExtensionCommand;
class WebExtensionContext;
struct WebExtensionMenuItemContextParameters;
struct WebExtensionMenuItemParameters;

class WebExtensionMenuItem : public RefCountedAndCanMakeWeakPtr<WebExtensionMenuItem> {
    WTF_MAKE_NONCOPYABLE(WebExtensionMenuItem);
    WTF_MAKE_TZONE_ALLOCATED(WebExtensionMenuItem);

public:
    template<typename... Args>
    static Ref<WebExtensionMenuItem> create(Args&&... args)
    {
        return adoptRef(*new WebExtensionMenuItem(std::forward<Args>(args)...));
    }

    using MenuItemVector = Vector<Ref<WebExtensionMenuItem>>;

    static NSArray *matchingPlatformMenuItems(const MenuItemVector&, const WebExtensionMenuItemContextParameters&, size_t limit = 0);

    bool operator==(const WebExtensionMenuItem&) const;

    WebExtensionMenuItemParameters minimalParameters() const;

    WebExtensionContext* extensionContext() const;

    bool matches(const WebExtensionMenuItemContextParameters&) const;

    void update(const WebExtensionMenuItemParameters&);

    WebExtensionMenuItemType type() const { return m_type; }
    const String& identifier() const { return m_identifier; }
    const String& title() const { return m_title; }

    WebExtensionCommand* command() const { return m_command.get(); }

    RefPtr<WebCore::Icon> icon(WebCore::FloatSize) const;

    bool isChecked() const { return m_checked; }
    void setChecked(bool checked) { ASSERT(isCheckedType(type())); m_checked = checked; }

    bool toggleCheckedIfNeeded(const WebExtensionMenuItemContextParameters&);

    bool isEnabled() const { return m_enabled; }
    bool isVisible() const { return m_visible; }

    const WebExtension::MatchPatternSet& documentPatterns() const { return m_documentPatterns; }
    const WebExtension::MatchPatternSet& targetPatterns() const { return m_targetPatterns; }
    OptionSet<WebExtensionMenuItemContextType> contexts() const { return m_contexts; }

    WebExtensionMenuItem* parentMenuItem() const { return m_parentMenuItem.get(); }
    const MenuItemVector& submenuItems() const { return m_submenuItems; }

    void addSubmenuItem(WebExtensionMenuItem&);
    void removeSubmenuItem(WebExtensionMenuItem&);

private:
    explicit WebExtensionMenuItem(WebExtensionContext&, const WebExtensionMenuItemParameters&);

    static String removeAmpersands(const String&);

    void clearIconCache() const;

    enum class ForceUnchecked : bool { No, Yes };
    CocoaMenuItem *platformMenuItem(const WebExtensionMenuItemContextParameters&, ForceUnchecked = ForceUnchecked::No) const;

    WeakPtr<WebExtensionContext> m_extensionContext;

    WebExtensionMenuItemType m_type;
    String m_identifier;
    String m_title;

    RefPtr<WebExtensionCommand> m_command;

    mutable RefPtr<WebCore::Icon> m_cachedIcon;
    mutable Vector<double> m_cachedIconScales;
    mutable WebCore::FloatSize m_cachedIconIdealSize;

    RefPtr<JSON::Object> m_icons;
#if ENABLE(WK_WEB_EXTENSIONS_ICON_VARIANTS)
    RefPtr<JSON::Array> m_iconVariants;
#endif

    bool m_checked : 1 { false };
    bool m_enabled : 1 { true };
    bool m_visible : 1 { true };

    WebExtension::MatchPatternSet m_documentPatterns;
    WebExtension::MatchPatternSet m_targetPatterns;
    OptionSet<WebExtensionMenuItemContextType> m_contexts;

    WeakPtr<WebExtensionMenuItem> m_parentMenuItem;
    MenuItemVector m_submenuItems;
};

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
