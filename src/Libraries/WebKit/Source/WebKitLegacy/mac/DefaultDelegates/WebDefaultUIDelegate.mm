/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 19, 2025.
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
#import "WebDefaultUIDelegate.h"

#import "WebUIDelegatePrivate.h"
#import "WebView.h"
#import <wtf/NeverDestroyed.h>
#import <wtf/RetainPtr.h>

#if PLATFORM(MAC)
#import "WebJavaScriptTextInputPanel.h"
#import <wtf/cocoa/RuntimeApplicationChecksCocoa.h>
#endif

#if PLATFORM(IOS_FAMILY)
#import <WebCore/WAKWindow.h>
#import <WebCore/WKViewPrivate.h>
#endif

#if PLATFORM(MAC)
@interface NSApplication (DeclarationStolenFromAppKit)
- (void)_cycleWindowsReversed:(BOOL)reversed;
@end
#endif

@implementation WebDefaultUIDelegate

// Return a object with vanilla implementations of the protocol's methods
// Note this feature relies on our default delegate being stateless. This
// is probably an invalid assumption for the WebUIDelegate.
// If we add any real functionality to this default delegate we probably
// won't be able to use a singleton.
+ (WebDefaultUIDelegate *)sharedUIDelegate
{
    static NeverDestroyed<RetainPtr<WebDefaultUIDelegate>> sharedDelegate = adoptNS([[WebDefaultUIDelegate alloc] init]);
    return sharedDelegate.get().get();
}

- (WebView *)webView: (WebView *)wv createWebViewWithRequest:(NSURLRequest *)request windowFeatures:(NSDictionary *)features
{
    // If the new API method doesn't exist, fallback to the old version of createWebViewWithRequest
    // for backwards compatability
    if (![[wv UIDelegate] respondsToSelector:@selector(webView:createWebViewWithRequest:windowFeatures:)] && [[wv UIDelegate] respondsToSelector:@selector(webView:createWebViewWithRequest:)])
        return [[wv UIDelegate] webView:wv createWebViewWithRequest:request];
    return nil;
}

#if PLATFORM(IOS_FAMILY)
- (WebView *)webView:(WebView *)sender createWebViewWithRequest:(NSURLRequest *)request userGesture:(BOOL)userGesture
{
    return nil;
}
#endif

- (void)webViewShow: (WebView *)wv
{
}

- (void)webViewClose: (WebView *)wv
{
#if PLATFORM(MAC)
    [[wv window] close];
#endif
}

- (void)webViewFocus: (WebView *)wv
{
#if PLATFORM(MAC)
    [[wv window] makeKeyAndOrderFront:wv];
#endif
}

- (void)webViewUnfocus: (WebView *)wv
{
#if PLATFORM(MAC)
    if ([[wv window] isKeyWindow] || [[[wv window] attachedSheet] isKeyWindow])
        [NSApp _cycleWindowsReversed:FALSE];
#endif
}

- (NSResponder *)webViewFirstResponder: (WebView *)wv
{
    return [[wv window] firstResponder];
}

- (void)webView: (WebView *)wv makeFirstResponder:(NSResponder *)responder
{
    [[wv window] makeFirstResponder:responder];
}

- (void)webView: (WebView *)wv setStatusText:(NSString *)text
{
}

- (NSString *)webViewStatusText: (WebView *)wv
{
    return nil;
}

- (void)webView: (WebView *)wv mouseDidMoveOverElement:(NSDictionary *)elementInformation modifierFlags:(NSUInteger)modifierFlags
{
}

- (BOOL)webViewAreToolbarsVisible: (WebView *)wv
{
    return NO;
}

- (void)webView: (WebView *)wv setToolbarsVisible:(BOOL)visible
{
}

- (BOOL)webViewIsStatusBarVisible: (WebView *)wv
{
    return NO;
}

- (void)webView: (WebView *)wv setStatusBarVisible:(BOOL)visible
{
}

- (BOOL)webViewIsResizable: (WebView *)wv
{
#if PLATFORM(IOS_FAMILY)
    return NO;
#else
    ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    return [[wv window] showsResizeIndicator];
    ALLOW_DEPRECATED_DECLARATIONS_END
#endif
}

- (void)webView: (WebView *)wv setResizable:(BOOL)resizable
{
#if PLATFORM(MAC)
    // FIXME: This doesn't actually change the resizability of the window,
    // only visibility of the indicator.
    ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    [[wv window] setShowsResizeIndicator:resizable];
    ALLOW_DEPRECATED_DECLARATIONS_END
#endif
}

- (void)webView: (WebView *)wv setFrame:(NSRect)frame
{
#if PLATFORM(MAC)
    [[wv window] setFrame:frame display:YES];
#endif
}

- (NSRect)webViewFrame: (WebView *)wv
{
    NSWindow *window = [wv window];
    return window == nil ? NSZeroRect : [window frame];
}

- (void)webView: (WebView *)wv runJavaScriptAlertPanelWithMessage:(NSString *)message initiatedByFrame:(WebFrame *)frame
{
    // FIXME: We want a default here, but that would add localized strings.
}

- (BOOL)webView: (WebView *)wv runJavaScriptConfirmPanelWithMessage:(NSString *)message initiatedByFrame:(WebFrame *)frame
{
    // FIXME: We want a default here, but that would add localized strings.
    return NO;
}

- (NSString *)webView: (WebView *)wv runJavaScriptTextInputPanelWithPrompt:(NSString *)prompt defaultText:(NSString *)defaultText initiatedByFrame:(WebFrame *)frame
{
#if PLATFORM(MAC)
    auto panel = adoptNS([[WebJavaScriptTextInputPanel alloc] initWithPrompt:prompt text:defaultText]);
    [panel showWindow:nil];
    NSString *result;
    if ([NSApp runModalForWindow:[panel window]])
        result = [panel text];
    else
        result = nil;
    [[panel window] close];
    return result;
#else
    return nil;
#endif
}

- (void)webView: (WebView *)wv runOpenPanelForFileButtonWithResultListener:(id<WebOpenPanelResultListener>)resultListener
{
    // FIXME: We want a default here, but that would add localized strings.
}

- (void)webView:(WebView *)sender printFrameView:(WebFrameView *)frameView
{
}

#if PLATFORM(MAC)
- (NSUInteger)webView:(WebView *)webView dragDestinationActionMaskForDraggingInfo:(id <NSDraggingInfo>)draggingInfo
{
    if (!linkedOnOrAfterSDKWithBehavior(SDKAlignedBehavior::DropToNavigateDisallowedByDefault))
        return WebDragDestinationActionAny;

    return WebDragDestinationActionAny & ~WebDragDestinationActionLoad;
}

- (void)webView:(WebView *)webView willPerformDragDestinationAction:(WebDragDestinationAction)action forDraggingInfo:(id <NSDraggingInfo>)draggingInfo
{
}

- (NSUInteger)webView:(WebView *)webView dragSourceActionMaskForPoint:(NSPoint)point
{
    return WebDragSourceActionAny;
}

- (void)webView:(WebView *)webView willPerformDragSourceAction:(WebDragSourceAction)action fromPoint:(NSPoint)point withPasteboard:(NSPasteboard *)pasteboard
{
}

- (void)webView:(WebView *)sender willPopupMenu:(NSMenu *)menu
{
}
#endif

- (void)webView:(WebView *)sender didDrawRect:(NSRect)rect
{
}

- (void)webView:(WebView *)sender didScrollDocumentInFrameView:(WebFrameView *)frameView
{
}

- (void)webView:(WebView *)sender exceededApplicationCacheOriginQuotaForSecurityOrigin:(WebSecurityOrigin *)origin totalSpaceNeeded:(NSUInteger)totalSpaceNeeded
{
}

#if PLATFORM(IOS_FAMILY)
- (void)webViewSupportedOrientationsUpdated:(WebView *)sender
{
}

#if ENABLE(DRAG_SUPPORT)
- (WebDragDestinationAction)webView:(WebView *)sender dragDestinationActionMaskForSession:(id <UIDropSession>)session
{
    return WebDragDestinationActionAny & ~WebDragDestinationActionLoad;
}
#endif
#endif

@end
