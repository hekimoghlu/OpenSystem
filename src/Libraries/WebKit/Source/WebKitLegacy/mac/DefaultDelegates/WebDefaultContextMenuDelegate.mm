/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 20, 2023.
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
#if !PLATFORM(IOS_FAMILY)

#import "WebDefaultContextMenuDelegate.h"

#import "WebDOMOperations.h"
#import "WebDataSourcePrivate.h"
#import "WebDefaultUIDelegate.h"
#import "WebFrameInternal.h"
#import "WebFrameView.h"
#import "WebHTMLViewInternal.h"
#import "WebLocalizableStringsInternal.h"
#import "WebNSPasteboardExtras.h"
#import "WebNSURLRequestExtras.h"
#import "WebPolicyDelegate.h"
#import "WebUIDelegate.h"
#import "WebUIDelegatePrivate.h"
#import "WebViewInternal.h"
#import <Foundation/NSURLConnection.h>
#import <Foundation/NSURLRequest.h>
#import <WebCore/Editor.h>
#import <WebCore/FrameLoader.h>
#import <WebCore/LocalFrame.h>
#import <WebKitLegacy/DOM.h>
#import <WebKitLegacy/DOMPrivate.h>
#import <pal/system/mac/DefaultSearchProvider.h>
#import <wtf/Assertions.h>

@implementation WebDefaultUIDelegate (WebContextMenu)

- (NSMenuItem *)menuItemWithTag:(int)tag target:(id)target representedObject:(id)representedObject
{
    auto menuItem = adoptNS([[NSMenuItem alloc] init]);
    [menuItem setTag:tag];
    [menuItem setTarget:target]; // can be nil
    [menuItem setRepresentedObject:representedObject];
    
    NSString *title = nil;
    SEL action = NULL;
    
    switch(tag) {
        case WebMenuItemTagCopy:
            title = UI_STRING_INTERNAL("Copy", "Copy context menu item");
            action = @selector(copy:);
            break;
        case WebMenuItemTagGoBack:
            title = UI_STRING_INTERNAL("Back", "Back context menu item");
            action = @selector(goBack:);
            break;
        case WebMenuItemTagGoForward:
            title = UI_STRING_INTERNAL("Forward", "Forward context menu item");
            action = @selector(goForward:);
            break;
        case WebMenuItemTagStop:
            title = UI_STRING_INTERNAL("Stop", "Stop context menu item");
            action = @selector(stopLoading:);
            break;
        case WebMenuItemTagReload:
            title = UI_STRING_INTERNAL("Reload", "Reload context menu item");
            action = @selector(reload:);
            break;
        case WebMenuItemTagSearchWeb: {
            auto searchProviderName = PAL::defaultSearchProviderDisplayName();
            title = [NSString stringWithFormat:UI_STRING_INTERNAL("Search with %@", "Search with search provider context menu item with provider name inserted"), searchProviderName.get()];
            action = @selector(_searchWithGoogleFromMenu:);
            break;
        }
        case WebMenuItemTagLookUpInDictionary:
            title = UI_STRING_INTERNAL("Look Up in Dictionary", "Look Up in Dictionary context menu item");
            action = @selector(_lookUpInDictionaryFromMenu:);
            break;
        case WebMenuItemTagOpenFrameInNewWindow:
            title = UI_STRING_INTERNAL("Open Frame in New Window", "Open Frame in New Window context menu item");
            action = @selector(_openFrameInNewWindowFromMenu:);
            break;
        default:
            ASSERT_NOT_REACHED();
            return nil;
    }

    if (title)
        [menuItem setTitle:title];

    [menuItem setAction:action];
    
    return menuItem.autorelease();
}

- (void)appendDefaultItems:(NSArray *)defaultItems toArray:(NSMutableArray *)menuItems
{
    ASSERT_ARG(menuItems, menuItems != nil);
    if ([defaultItems count] > 0) {
        ASSERT(![[menuItems lastObject] isSeparatorItem]);
        if (![[defaultItems objectAtIndex:0] isSeparatorItem]) {
            [menuItems addObject:[NSMenuItem separatorItem]];
            
            NSEnumerator *e = [defaultItems objectEnumerator];
            NSMenuItem *item;
            while ((item = [e nextObject]) != nil) {
                [menuItems addObject:item];
            }
        }
    }
}

- (NSArray *)webView:(WebView *)wv contextMenuItemsForElement:(NSDictionary *)element  defaultMenuItems:(NSArray *)defaultMenuItems
{
    // The defaultMenuItems here are ones supplied by the WebDocumentView protocol implementation. WebPDFView is
    // one case that has non-nil default items here.
    NSMutableArray *menuItems = [NSMutableArray array];

    WebFrame *webFrame = [element objectForKey:WebElementFrameKey];
    
    if ([[element objectForKey:WebElementIsSelectedKey] boolValue]) {
        // The Google item is implemented in WebView, and requires that the
        // current document view conforms to WebDocumentText
        ASSERT([[[webFrame frameView] documentView] conformsToProtocol:@protocol(WebDocumentText)]);

        // FIXME 4184640: The Look Up in Dictionary item is only implemented in WebHTMLView, and so is present but
        // dimmed for other cases where WebElementIsSelectedKey is present. It would probably 
        // be better not to include it in the menu if the documentView isn't a WebHTMLView, but that could break 
        // existing clients that have code that relies on it being present (unlikely for clients outside of Apple, 
        // but Safari has such code).

        NSMenuItem *lookupMenuItem = [self menuItemWithTag:WebMenuItemTagLookUpInDictionary target:nil representedObject:element];
        NSString *selectedString = [(id <WebDocumentText>)[[webFrame frameView] documentView] selectedString];
        [lookupMenuItem setTitle:[NSString stringWithFormat:UI_STRING_INTERNAL("Look Up â€œ%@â€", "Look Up context menu item with selected word"), selectedString]];
        [menuItems addObject:lookupMenuItem];

        [menuItems addObject:[self menuItemWithTag:WebMenuItemTagSearchWeb target:nil representedObject:element]];

        [menuItems addObject:[NSMenuItem separatorItem]];
        [menuItems addObject:[self menuItemWithTag:WebMenuItemTagCopy target:nil representedObject:element]];
    } else {
        WebView *wv = [webFrame webView];
        if ([wv canGoBack]) {
            [menuItems addObject:[self menuItemWithTag:WebMenuItemTagGoBack target:wv representedObject:element]];
        }
        if ([wv canGoForward]) {
            [menuItems addObject:[self menuItemWithTag:WebMenuItemTagGoForward target:wv representedObject:element]];
        }
        if ([wv isLoading]) {
            [menuItems addObject:[self menuItemWithTag:WebMenuItemTagStop target:wv representedObject:element]];
        } else {
            [menuItems addObject:[self menuItemWithTag:WebMenuItemTagReload target:wv representedObject:element]];
        }

        if (webFrame != [wv mainFrame]) {
            [menuItems addObject:[self menuItemWithTag:WebMenuItemTagOpenFrameInNewWindow target:wv representedObject:element]];
        }
    }
    
    // Add the default items at the end, if any, after a separator
    [self appendDefaultItems:defaultMenuItems toArray:menuItems];

    return menuItems;
}

@end

#endif // !PLATFORM(IOS_FAMILY)
