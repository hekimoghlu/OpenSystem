/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 13, 2024.
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
#if !__has_feature(objc_arc)
#error This file requires ARC. Add the "-fobjc-arc" compiler flag for this file.
#endif

#import "config.h"
#import "WebExtensionCommand.h"

#if ENABLE(WK_WEB_EXTENSIONS)

#import "CocoaHelpers.h"
#import "WebExtension.h"
#import "WebExtensionContext.h"
#import "WebExtensionMenuItem.h"
#import <wtf/BlockPtr.h>
#import <wtf/text/StringBuilder.h>

#if USE(APPKIT)
#import "AppKitSPI.h"
#import <Carbon/Carbon.h>
#endif

#if PLATFORM(IOS_FAMILY)
#import <UIKit/UIKit.h>
#endif

#if PLATFORM(IOS_FAMILY)
@implementation _WKWebExtensionKeyCommand

+ (UIKeyCommand *)commandWithTitle:(NSString *)title image:(UIImage *)image input:(NSString *)input modifierFlags:(UIKeyModifierFlags)modifierFlags identifier:(NSString *)identifier
{
    RELEASE_ASSERT(title);
    RELEASE_ASSERT(input);
    RELEASE_ASSERT(identifier);

    auto *propertyList = @{
        @"title": title,
        @"activation": input,
        @"identifier": identifier,
    };

    auto *command = [UIKeyCommand commandWithTitle:title image:image action:@selector(performWebExtensionCommandForKeyCommand:) input:input modifierFlags:modifierFlags propertyList:propertyList];
    if (!command)
        return nil;

    return command;
}

- (void)performWebExtensionCommandForKeyCommand:(id)sender
{
    // To be handled by the first responder instead of this class.
    // This method exists just to make this symbol available.

    // FIXME: Would put ASSERT_NOT_REACHED() here but some compilers are warning the function is "noreturn".
}

@end
#endif // PLATFORM(IOS_FAMILY)

namespace WebKit {

void WebExtensionCommand::dispatchChangedEventSoonIfNeeded()
{
    // Already scheduled to fire if old shortcut is set.
    if (!m_oldShortcut.isNull())
        return;

    m_oldShortcut = shortcutString();

    dispatch_async(dispatch_get_main_queue(), makeBlockPtr([this, protectedThis = Ref { *this }]() {
        RefPtr context = extensionContext();
        if (!context)
            return;

        context->fireCommandChangedEventIfNeeded(*this, m_oldShortcut);

        m_oldShortcut = nullString();
    }).get());
}

bool WebExtensionCommand::setActivationKey(String activationKey)
{
    if (activationKey.isEmpty()) {
        m_activationKey = nullString();
        return true;
    }

    if (activationKey.length() > 1)
        return false;

    static NSCharacterSet *notAllowedCharacterSet;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        auto *allowedCharacterSet = [NSMutableCharacterSet alphanumericCharacterSet];
        // F1-F12.
        [allowedCharacterSet addCharactersInRange:NSMakeRange(0xF704, 12)];
        // Insert, Delete, Home.
        [allowedCharacterSet addCharactersInRange:NSMakeRange(0xF727, 3)];
        // End, Page Up, Page Down.
        [allowedCharacterSet addCharactersInRange:NSMakeRange(0xF72B, 3)];
        // Up, Down, Left, Right.
        [allowedCharacterSet addCharactersInRange:NSMakeRange(0xF700, 4)];
        [allowedCharacterSet addCharactersInString:@",. "];

        notAllowedCharacterSet = allowedCharacterSet.invertedSet;
    });

    if ([(NSString *)activationKey rangeOfCharacterFromSet:notAllowedCharacterSet].location != NSNotFound)
        return false;

    dispatchChangedEventSoonIfNeeded();

    m_activationKey = activationKey.convertToASCIILowercase();

    return true;
}

String WebExtensionCommand::shortcutString() const
{
    auto flags = modifierFlags();
    auto key = activationKey();

    if (!flags || key.isEmpty())
        return emptyString();

    static NeverDestroyed<HashMap<String, String>> specialKeyMap = HashMap<String, String> {
        { ","_s, "Comma"_s },
        { "."_s, "Period"_s },
        { " "_s, "Space"_s },
        { @"\uF704", "F1"_s },
        { @"\uF705", "F2"_s },
        { @"\uF706", "F3"_s },
        { @"\uF707", "F4"_s },
        { @"\uF708", "F5"_s },
        { @"\uF709", "F6"_s },
        { @"\uF70A", "F7"_s },
        { @"\uF70B", "F8"_s },
        { @"\uF70C", "F9"_s },
        { @"\uF70D", "F10"_s },
        { @"\uF70E", "F11"_s },
        { @"\uF70F", "F12"_s },
        { @"\uF727", "Insert"_s },
        { @"\uF728", "Delete"_s },
        { @"\uF729", "Home"_s },
        { @"\uF72B", "End"_s },
        { @"\uF72C", "PageUp"_s },
        { @"\uF72D", "PageDown"_s },
        { @"\uF700", "Up"_s },
        { @"\uF701", "Down"_s },
        { @"\uF702", "Left"_s },
        { @"\uF703", "Right"_s }
    };

    StringBuilder stringBuilder;

    if (flags.contains(ModifierFlags::Control))
        stringBuilder.append("MacCtrl"_s);

    if (flags.contains(ModifierFlags::Option)) {
        if (!stringBuilder.isEmpty())
            stringBuilder.append('+');
        stringBuilder.append("Alt"_s);
    }

    if (flags.contains(ModifierFlags::Shift)) {
        if (!stringBuilder.isEmpty())
            stringBuilder.append('+');
        stringBuilder.append("Shift"_s);
    }

    if (flags.contains(ModifierFlags::Command)) {
        if (!stringBuilder.isEmpty())
            stringBuilder.append('+');
        stringBuilder.append("Command"_s);
    }

    if (!stringBuilder.isEmpty())
        stringBuilder.append('+');

    if (auto specialKey = specialKeyMap.get().get(key); !specialKey.isEmpty())
        stringBuilder.append(specialKey);
    else
        stringBuilder.append(key.convertToASCIIUppercase());

    return stringBuilder.toString();
}

String WebExtensionCommand::userVisibleShortcut() const
{
    auto flags = modifierFlags();
    auto key = activationKey();

    if (!flags || key.isEmpty())
        return emptyString();

#if PLATFORM(MAC)
    NSKeyboardShortcut *shortcut = [NSKeyboardShortcut shortcutWithKeyEquivalent:key modifierMask:flags.toRaw()];
    return shortcut.localizedDisplayName ?: @"";
#else
    static NeverDestroyed<HashMap<String, String>> specialKeyMap = HashMap<String, String> {
        { ","_s, ","_s },
        { "."_s, "."_s },
        { " "_s, "Space"_s },
        { @"\uF704", "F1"_s },
        { @"\uF705", "F2"_s },
        { @"\uF706", "F3"_s },
        { @"\uF707", "F4"_s },
        { @"\uF708", "F5"_s },
        { @"\uF709", "F6"_s },
        { @"\uF70A", "F7"_s },
        { @"\uF70B", "F8"_s },
        { @"\uF70C", "F9"_s },
        { @"\uF70D", "F10"_s },
        { @"\uF70E", "F11"_s },
        { @"\uF70F", "F12"_s },
        { @"\uF727", "Ins"_s },
        { @"\uF728", @"âŒ«" },
        { @"\uF729", @"â†–ï¸Ž" },
        { @"\uF72B", @"â†˜ï¸Ž" },
        { @"\uF72C", @"â‡ž" },
        { @"\uF72D", @"â‡Ÿ" },
        { @"\uF700", @"â†‘" },
        { @"\uF701", @"â†“" },
        { @"\uF702", @"â†" },
        { @"\uF703", @"â†’" }
    };

    StringBuilder stringBuilder;

    if (flags.contains(ModifierFlags::Control))
        stringBuilder.append(String(@"âŒƒ"));

    if (flags.contains(ModifierFlags::Option))
        stringBuilder.append(String(@"âŒ¥"));

    if (flags.contains(ModifierFlags::Shift))
        stringBuilder.append(String(@"â‡§"));

    if (flags.contains(ModifierFlags::Command))
        stringBuilder.append(String(@"âŒ˜"));

    if (auto specialKey = specialKeyMap.get().get(key); !specialKey.isEmpty())
        stringBuilder.append(specialKey);
    else
        stringBuilder.append(key.convertToASCIIUppercase());

    return stringBuilder.toString();
#endif
}

CocoaMenuItem *WebExtensionCommand::platformMenuItem() const
{
#if USE(APPKIT)
    auto *result = [[_WKWebExtensionMenuItem alloc] initWithTitle:description() handler:makeBlockPtr([this, protectedThis = Ref { *this }](id sender) mutable {
        if (RefPtr context = extensionContext())
            context->performCommand(const_cast<WebExtensionCommand&>(*this), WebExtensionContext::UserTriggered::Yes);
    }).get()];

    result.keyEquivalent = activationKey();
    result.keyEquivalentModifierMask = modifierFlags().toRaw();
    if (RefPtr context = extensionContext())
        result.image = toCocoaImage(context->extension().icon(WebCore::FloatSize(16, 16)));

    return result;
#else
    return [UIAction actionWithTitle:description() image:nil identifier:nil handler:makeBlockPtr([this, protectedThis = Ref { *this }](UIAction *) mutable {
        if (RefPtr context = extensionContext())
            context->performCommand(const_cast<WebExtensionCommand&>(*this), WebExtensionContext::UserTriggered::Yes);
    }).get()];
#endif
}

#if PLATFORM(IOS_FAMILY)
UIKeyCommand *WebExtensionCommand::keyCommand() const
{
    if (activationKey().isEmpty())
        return nil;

    return [_WKWebExtensionKeyCommand commandWithTitle:description() image:nil input:activationKey() modifierFlags:modifierFlags().toRaw() identifier:identifier()];
}

bool WebExtensionCommand::matchesKeyCommand(UIKeyCommand *keyCommand) const
{
    return keyCommand.modifierFlags == modifierFlags().toRaw() && [keyCommand.input isEqual:activationKey()] && [keyCommand.propertyList[@"identifier"] isEqual:identifier()];
}
#endif

#if USE(APPKIT)
bool WebExtensionCommand::matchesEvent(NSEvent *event) const
{
    if (event.type != NSEventTypeKeyDown || event.isARepeat)
        return false;

    if (activationKey().isEmpty())
        return false;

    auto expectedModifierFlags = modifierFlags().toRaw();
    if ((event.modifierFlags & expectedModifierFlags) != expectedModifierFlags)
        return false;

    static NeverDestroyed<HashMap<String, uint16_t>> specialKeyMap = HashMap<String, uint16_t> {
        { ","_s, kVK_ANSI_Comma },
        { "."_s, kVK_ANSI_Period },
        { " "_s, kVK_Space },
        { @"\uF704", kVK_F1 },
        { @"\uF705", kVK_F2 },
        { @"\uF706", kVK_F3 },
        { @"\uF707", kVK_F4 },
        { @"\uF708", kVK_F5 },
        { @"\uF709", kVK_F6 },
        { @"\uF70A", kVK_F7 },
        { @"\uF70B", kVK_F8 },
        { @"\uF70C", kVK_F9 },
        { @"\uF70D", kVK_F10 },
        { @"\uF70E", kVK_F11 },
        { @"\uF70F", kVK_F12 },
        // Insert (\uF727) is not present on Apple keyboards.
        { @"\uF728", kVK_ForwardDelete },
        { @"\uF729", kVK_Home },
        { @"\uF72B", kVK_End },
        { @"\uF72C", kVK_PageUp },
        { @"\uF72D", kVK_PageDown },
        { @"\uF700", kVK_UpArrow },
        { @"\uF701", kVK_DownArrow },
        { @"\uF702", kVK_LeftArrow },
        { @"\uF703", kVK_RightArrow }
    };

    if (equalIgnoringASCIICase(activationKey(), String(event.charactersIgnoringModifiers)))
        return true;

    if (auto mappedKeyCode = specialKeyMap.get().get(activationKey()))
        return mappedKeyCode == event.keyCode;

    return false;
}
#endif // USE(APPKIT)

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
