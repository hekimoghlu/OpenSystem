/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 22, 2024.
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
#import "WKWebExtensionCommandInternal.h"

#import "WebExtensionCommand.h"
#import "WebExtensionContext.h"

#if USE(APPKIT)
using CocoaModifierFlags = NSEventModifierFlags;
using CocoaMenuItem = NSMenuItem;
#else
using CocoaModifierFlags = UIKeyModifierFlags;
using CocoaMenuItem = UIMenuElement;
#endif

@implementation WKWebExtensionCommand

#if ENABLE(WK_WEB_EXTENSIONS)

WK_OBJECT_DEALLOC_IMPL_ON_MAIN_THREAD(WKWebExtensionCommand, WebExtensionCommand, _webExtensionCommand);

- (NSUInteger)hash
{
    return self.identifier.hash;
}

- (BOOL)isEqual:(id)object
{
    if (self == object)
        return YES;

    auto *other = dynamic_objc_cast<WKWebExtensionCommand>(object);
    if (!other)
        return NO;

    return *_webExtensionCommand == *other->_webExtensionCommand;
}

- (NSString *)description
{
    return self.title;
}

- (NSString *)debugDescription
{
    return [NSString stringWithFormat:@"<%@: %p; identifier = %@; shortcut = %@>", NSStringFromClass(self.class), self,
        self.identifier, self.activationKey.length ? (NSString *)self._protectedWebExtensionCommand->shortcutString() : @"(none)"];
}

- (WKWebExtensionContext *)webExtensionContext
{
    if (RefPtr context = self._protectedWebExtensionCommand->extensionContext())
        return context->wrapper();
    return nil;
}

- (NSString *)identifier
{
    return _webExtensionCommand->identifier();
}

- (NSString *)title
{
    return _webExtensionCommand->description();
}

- (NSString *)activationKey
{
    if (auto& activationKey = self._protectedWebExtensionCommand->activationKey(); !activationKey.isEmpty())
        return activationKey;
    return nil;
}

- (void)setActivationKey:(NSString *)activationKey
{
    bool result = self._protectedWebExtensionCommand->setActivationKey(activationKey);
    NSAssert(result, @"Invalid parameter: an unsupported character was provided");
}

- (CocoaModifierFlags)modifierFlags
{
    return self._protectedWebExtensionCommand->modifierFlags().toRaw();
}

- (void)setModifierFlags:(CocoaModifierFlags)modifierFlags
{
    auto optionSet = OptionSet<WebKit::WebExtension::ModifierFlags>::fromRaw(modifierFlags) & WebKit::WebExtension::allModifierFlags();
    NSAssert(optionSet.toRaw() == modifierFlags, @"Invalid parameter: an unsupported modifier flag was provided");

    self._protectedWebExtensionCommand->setModifierFlags(optionSet);
}

- (CocoaMenuItem *)menuItem
{
    return self._protectedWebExtensionCommand->platformMenuItem();
}

#if PLATFORM(IOS_FAMILY)
- (UIKeyCommand *)keyCommand
{
    return _webExtensionCommand->keyCommand();
}
#endif

- (NSString *)_shortcut
{
    return self._protectedWebExtensionCommand->shortcutString();
}

- (NSString *)_userVisibleShortcut
{
    return self._protectedWebExtensionCommand->userVisibleShortcut();
}

- (BOOL)_isActionCommand
{
    return self._protectedWebExtensionCommand->isActionCommand();
}

#if USE(APPKIT)
- (BOOL)_matchesEvent:(NSEvent *)event
{
    NSParameterAssert([event isKindOfClass:NSEvent.class]);

    return self._protectedWebExtensionCommand->matchesEvent(event);
}
#endif

#pragma mark WKObject protocol implementation

- (API::Object&)_apiObject
{
    return *_webExtensionCommand;
}

- (WebKit::WebExtensionCommand&)_webExtensionCommand
{
    return *_webExtensionCommand;
}

- (Ref<WebKit::WebExtensionCommand>)_protectedWebExtensionCommand
{
    return *_webExtensionCommand;
}

#else // ENABLE(WK_WEB_EXTENSIONS)

- (WKWebExtensionContext *)webExtensionContext
{
    return nil;
}

- (NSString *)identifier
{
    return nil;
}

- (NSString *)title
{
    return nil;
}

- (NSString *)activationKey
{
    return nil;
}

- (void)setActivationKey:(NSString *)activationKey
{
}

- (CocoaModifierFlags)modifierFlags
{
    return 0;
}

- (void)setModifierFlags:(CocoaModifierFlags)modifierFlags
{
}

- (CocoaMenuItem *)menuItem
{
    return nil;
}

#if PLATFORM(IOS_FAMILY)
- (UIKeyCommand *)keyCommand
{
    return nil;
}
#endif

- (NSString *)_shortcut
{
    return nil;
}

- (NSString *)_userVisibleShortcut
{
    return nil;
}

- (BOOL)_isActionCommand
{
    return NO;
}

#if USE(APPKIT)
- (BOOL)_matchesEvent:(NSEvent *)event
{
    return NO;
}
#endif

#endif // ENABLE(WK_WEB_EXTENSIONS)

@end
