/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 3, 2023.
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
#import "WKWebExtensionActionInternal.h"

#import "CocoaHelpers.h"
#import "CocoaImage.h"
#import "WebExtensionAction.h"
#import "WebExtensionContext.h"
#import "WebExtensionTab.h"
#import <wtf/BlockPtr.h>
#import <wtf/CompletionHandler.h>

#if USE(APPKIT)
using CocoaMenuItem = NSMenuItem;
#else
using CocoaMenuItem = UIMenuElement;
#endif

@implementation WKWebExtensionAction

#if ENABLE(WK_WEB_EXTENSIONS)

WK_OBJECT_DEALLOC_IMPL_ON_MAIN_THREAD(WKWebExtensionAction, WebExtensionAction, _webExtensionAction);

- (BOOL)isEqual:(id)object
{
    if (self == object)
        return YES;

    auto *other = dynamic_objc_cast<WKWebExtensionAction>(object);
    if (!other)
        return NO;

    return *_webExtensionAction == *other->_webExtensionAction;
}

- (WKWebExtensionContext *)webExtensionContext
{
    if (RefPtr context = self._protectedWebExtensionAction->extensionContext())
        return context->wrapper();
    return nil;
}

- (id<WKWebExtensionTab>)associatedTab
{
    if (RefPtr tab = self._protectedWebExtensionAction->tab())
        return tab->delegate();
    return nil;
}

- (CocoaImage *)iconForSize:(CGSize)size
{
    return WebKit::toCocoaImage(self._protectedWebExtensionAction->icon(WebCore::FloatSize(size)));
}

- (NSString *)label
{
    return self._protectedWebExtensionAction->label();
}

- (NSString *)badgeText
{
    return self._protectedWebExtensionAction->badgeText();
}

- (BOOL)hasUnreadBadgeText
{
    return self._protectedWebExtensionAction->hasUnreadBadgeText();
}

- (void)setHasUnreadBadgeText:(BOOL)hasUnreadBadgeText
{
    return self._protectedWebExtensionAction->setHasUnreadBadgeText(hasUnreadBadgeText);
}

- (NSString *)inspectionName
{
    return self._protectedWebExtensionAction->popupWebViewInspectionName();
}

- (void)setInspectionName:(NSString *)name
{
    self._protectedWebExtensionAction->setPopupWebViewInspectionName(name);
}

- (BOOL)isEnabled
{
    return self._protectedWebExtensionAction->isEnabled();
}

- (NSArray<CocoaMenuItem *> *)menuItems
{
    return self._protectedWebExtensionAction->platformMenuItems();
}

- (BOOL)presentsPopup
{
    return self._protectedWebExtensionAction->presentsPopup();
}

#if PLATFORM(IOS_FAMILY)
- (UIViewController *)popupViewController
{
    return self._protectedWebExtensionAction->popupViewController();
}
#endif

#if PLATFORM(MAC)
- (NSPopover *)popupPopover
{
    return self._protectedWebExtensionAction->popupPopover();
}
#endif

- (WKWebView *)popupWebView
{
    return self._protectedWebExtensionAction->popupWebView();
}

- (void)closePopup
{
    self._protectedWebExtensionAction->closePopup();
}

#pragma mark WKObject protocol implementation

- (API::Object&)_apiObject
{
    return *_webExtensionAction;
}

- (WebKit::WebExtensionAction&)_webExtensionAction
{
    return *_webExtensionAction;
}

- (Ref<WebKit::WebExtensionAction>)_protectedWebExtensionAction
{
    return *_webExtensionAction;
}

#else // ENABLE(WK_WEB_EXTENSIONS)

- (WKWebExtensionContext *)webExtensionContext
{
    return nil;
}

- (id<WKWebExtensionTab>)associatedTab
{
    return nil;
}

- (CocoaImage *)iconForSize:(CGSize)size
{
    return nil;
}

- (NSString *)label
{
    return nil;
}

- (NSString *)badgeText
{
    return nil;
}

- (BOOL)hasUnreadBadgeText
{
    return NO;
}

- (void)setHasUnreadBadgeText:(BOOL)hasUnreadBadgeText
{
}

- (NSString *)inspectionName
{
    return nil;
}

- (void)setInspectionName:(NSString *)name
{
}

- (BOOL)isEnabled
{
    return NO;
}

- (NSArray<CocoaMenuItem *> *)menuItems
{
    return nil;
}

- (BOOL)presentsPopup
{
    return NO;
}

#if PLATFORM(IOS_FAMILY)
- (UIViewController *)popupViewController
{
    return nil;
}
#endif

#if PLATFORM(MAC)
- (NSPopover *)popupPopover
{
    return nil;
}
#endif

- (WKWebView *)popupWebView
{
    return nil;
}

- (void)closePopup
{
}

#endif // ENABLE(WK_WEB_EXTENSIONS)

@end
