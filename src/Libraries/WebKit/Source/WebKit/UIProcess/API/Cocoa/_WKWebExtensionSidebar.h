/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 11, 2025.
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
#import <Foundation/Foundation.h>
#import <WebKit/WKFoundation.h>

@class WKWebExtensionContext;
@class WKWebView;
@protocol WKWebExtensionTab;
@protocol WKWebExtension;

#if TARGET_OS_IPHONE
@class UIImage;
@class UIViewController;
#else
@class NSImage;
@class NSViewController;
#endif

WK_HEADER_AUDIT_BEGIN(nullability, sendability)

/*!
 @abstract A `_WKWebExtensionSidebar` object encapsulates the properties for a specific web extension sidebar.
 @discussion When this property is `nil`, it indicates that the action is the default action and not associated with a specific tab.
 */
WK_CLASS_AVAILABLE(macos(15.2), ios(18.2), visionos(2.2))
NS_SWIFT_NAME(WKWebExtension.Sidebar)
@interface _WKWebExtensionSidebar : NSObject

- (instancetype)init NS_UNAVAILABLE;
- (instancetype)new NS_UNAVAILABLE;

/*! @abstract The extension context to which this sidebar is related. */
@property (nonatomic, nullable, readonly, weak) WKWebExtensionContext *webExtensionContext;

/*! @abstract The tab that this sidebar is associated with, or `nil` if it is the default sidebar */
@property (nonatomic, nullable, readonly, weak) id <WKWebExtensionTab> associatedTab;

/*! @abstract The title of this sidebar. */
@property (nonatomic, readonly, copy) NSString *title;

/*!
 @abstract Get the sidebar icon of the given size.
 @param size The size to use when looking up the sidebar icon.
 @result The sidebar icon, or the action icon if the sidebar specifies no icon, or `nil` if the action icon was unable to be loaded.
 */
#if TARGET_OS_IPHONE
- (nullable UIImage *)iconForSize:(CGSize)size;
#else
- (nullable NSImage *)iconForSize:(CGSize)size;
#endif

/*! @abstract Whether this sidebar is enabled or not. */
@property (nonatomic, readonly, getter=isEnabled) BOOL enabled;

/*! @abstract The web view which should be displayed when this sidebar is opened, or `nil` if this sidebar is not a tab-specific sidebar. */
@property (nonatomic, nullable, readonly) WKWebView *webView;

#if TARGET_OS_IPHONE
/*!
 @abstract A view controller that presents a web view which will load the sidebar page for this sidebar, or `nil` if this sidebar
 is not a tab-specific sidebar.
 */
@property (nonatomic, nullable, readonly) UIViewController *viewController;
#endif

#if TARGET_OS_OSX
/*!
 @abstract The view controller that presents a web view which will load the sidebar page for this sidebar, or `nil` if this sidebar
 is not a tab-specific sidebar.
 */
@property (nonatomic, nullable, readonly) NSViewController *viewController;
#endif

/*!
 @abstract Indicate that the sidebar will be opened
 @discussion This method should be invoked by the browser when this sidebar will be opened due to some action by the user. If
 this method is not called before the sidebar is opened, then the ``WKWebView`` associated with this sidebar may not have a
 document loaded, and the extension may not receive the `activeTab` permission from this user interaction.
 */
- (void)willOpenSidebar;

/*!
 @abstract Indicate that the sidebar will be closed
 @discussion This method should be invoked by the browser when the sidebar will be closed -- i.e., its associated ``WKWebView`` will cease
 to be displayed. If this method is not called when the sidebar is closed, then the sidebar's associated ``WKWebView`` may remain active longer than
 necessary. Note that calling this method does not guarantee that the ``WKWebView`` associated with a particular sidebar will be deallocated, as the
 web view may be shared between mutliple sidebars.
 */
- (void)willCloseSidebar;

@end // interface _WKWebExtensionSidebar

WK_HEADER_AUDIT_END(nullability, sendability)
