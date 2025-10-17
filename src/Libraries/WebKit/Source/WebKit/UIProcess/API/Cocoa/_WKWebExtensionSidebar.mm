/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 27, 2021.
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
#import "_WKWebExtensionSidebarInternal.h"

#import "CocoaHelpers.h"
#import "CocoaImage.h"
#import "WebExtensionContext.h"
#import "WebExtensionSidebar.h"
#import "WebExtensionTab.h"
#import <wtf/BlockPtr.h>
#import <wtf/CompletionHandler.h>

@implementation _WKWebExtensionSidebar

#if ENABLE(WK_WEB_EXTENSIONS_SIDEBAR)

WK_OBJECT_DEALLOC_IMPL_ON_MAIN_THREAD(_WKWebExtensionSidebar, WebExtensionSidebar, _webExtensionSidebar);

- (WKWebExtensionContext *)webExtensionContext
{
    return _webExtensionSidebar->extensionContext()
        .transform([](auto const& context) { return context->wrapper(); })
        .value_or(nil);
}

- (NSString *)title
{
    return _webExtensionSidebar->title();
}

- (CocoaImage *)iconForSize:(CGSize)size
{
    return WebKit::toCocoaImage(_webExtensionSidebar->icon(WebCore::FloatSize(size)));
}

- (SidebarViewControllerType *)viewController
{
    return _webExtensionSidebar->viewController().get();
}

- (BOOL)isEnabled
{
    return _webExtensionSidebar->isEnabled();
}

- (WKWebView *)webView
{
    return _webExtensionSidebar->webView();
}

- (void)willOpenSidebar
{
    _webExtensionSidebar->willOpenSidebar();
}

- (void)willCloseSidebar
{
    _webExtensionSidebar->willCloseSidebar();
}

- (id<WKWebExtensionTab>)associatedTab
{
    if (auto tab = _webExtensionSidebar->tab())
        return tab.value()->delegate();
    return nil;
}

- (API::Object&)_apiObject
{
    return *_webExtensionSidebar;
}

- (WebKit::WebExtensionSidebar&) _webExtensionSidebar
{
    return *_webExtensionSidebar;
}

#else // ENABLE(WK_WEB_EXTENSIONS_SIDEBAR)

- (WKWebExtensionContext *)webExtensionContext
{
    return nil;
}

- (NSString *)title
{
    return nil;
}

#if PLATFORM(MAC)
- (NSImage *)iconForSize:(CGSize)size
{
    return nil;
}
#endif

#if PLATFORM(IOS_FAMILY)
- (UIImage *)iconForSize:(CGSize)size
{
    return nil;
}
#endif

- (SidebarViewControllerType *)viewController
{
    return nil;
}

- (BOOL)isEnabled
{
    return false;
}

- (WKWebView *)webView
{
    return nil;
}

- (void)willOpenSidebar
{
}

- (void)willCloseSidebar
{
}

- (id<WKWebExtensionTab>)associatedTab
{
    return nil;
}

#endif // ENABLE(WK_WEB_EXTENSIONS_SIDEBAR)

@end
