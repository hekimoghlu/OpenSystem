/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 28, 2023.
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
#include "WebExtension.h"
#include <wtf/Forward.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

#if defined(__OBJC__) && PLATFORM(IOS_FAMILY)
#include <UIKit/UIKeyCommand.h>
#endif

OBJC_CLASS WKWebExtensionCommand;

#if USE(APPKIT)
OBJC_CLASS NSEvent;
OBJC_CLASS NSMenuItem;
using CocoaMenuItem = NSMenuItem;
#else
OBJC_CLASS UIKeyCommand;
OBJC_CLASS UIMenuElement;
using CocoaMenuItem = UIMenuElement;
#endif

#if defined(__OBJC__) && PLATFORM(IOS_FAMILY)
@interface _WKWebExtensionKeyCommand : UIKeyCommand

+ (UIKeyCommand *)commandWithTitle:(NSString *)title image:(UIImage *)image input:(NSString *)input modifierFlags:(UIKeyModifierFlags)modifierFlags identifier:(NSString *)identifier;

@end
#endif // PLATFORM(IOS_FAMILY)

namespace WebKit {

class WebExtensionContext;
struct WebExtensionCommandParameters;

class WebExtensionCommand : public API::ObjectImpl<API::Object::Type::WebExtensionCommand>, public CanMakeWeakPtr<WebExtensionCommand> {
    WTF_MAKE_NONCOPYABLE(WebExtensionCommand);

public:
    template<typename... Args>
    static Ref<WebExtensionCommand> create(Args&&... args)
    {
        return adoptRef(*new WebExtensionCommand(std::forward<Args>(args)...));
    }

    explicit WebExtensionCommand(WebExtensionContext&, const WebExtension::CommandData&);

    using ModifierFlags = WebExtension::ModifierFlags;

    bool operator==(const WebExtensionCommand&) const;

    WebExtensionCommandParameters parameters() const;

    WebExtensionContext* extensionContext() const;

    bool isActionCommand() const;

    const String& identifier() const { return m_identifier; }
    const String& description() const { return m_description; }

    const String& activationKey() const { return m_modifierFlags ? m_activationKey : nullString(); }
    bool setActivationKey(String);

    OptionSet<ModifierFlags> modifierFlags() const { return !m_activationKey.isEmpty() ? m_modifierFlags : OptionSet<ModifierFlags> { }; }
    void setModifierFlags(OptionSet<ModifierFlags> modifierFlags) { dispatchChangedEventSoonIfNeeded(); m_modifierFlags = modifierFlags; }

    String shortcutString() const;
    String userVisibleShortcut() const;

    CocoaMenuItem *platformMenuItem() const;

#if PLATFORM(IOS_FAMILY)
    UIKeyCommand *keyCommand() const;
    bool matchesKeyCommand(UIKeyCommand *) const;
#endif

#if USE(APPKIT)
    bool matchesEvent(NSEvent *) const;
#endif

#ifdef __OBJC__
    WKWebExtensionCommand *wrapper() const { return (WKWebExtensionCommand *)API::ObjectImpl<API::Object::Type::WebExtensionCommand>::wrapper(); }
#endif

private:
    void dispatchChangedEventSoonIfNeeded();

    WeakPtr<WebExtensionContext> m_extensionContext;
    String m_identifier;
    String m_description;
    String m_activationKey;
    OptionSet<ModifierFlags> m_modifierFlags;
    String m_oldShortcut;
};

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
