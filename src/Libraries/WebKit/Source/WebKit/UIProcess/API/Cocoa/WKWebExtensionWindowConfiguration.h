/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 28, 2025.
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

#import <WebKit/WKWebExtensionWindow.h>

@protocol WKWebExtensionTab;

WK_HEADER_AUDIT_BEGIN(nullability, sendability)

/*!
 @abstract A ``WKWebExtensionWindowConfiguration`` object encapsulates configuration options for a window in an extension.
 @discussion This class holds various options that influence the behavior and initial state of a window.
 The app retains the discretion to disregard any or all of these options, or even opt not to create a window.
 */
WK_CLASS_AVAILABLE(macos(WK_MAC_TBA), ios(WK_IOS_TBA), visionos(WK_XROS_TBA))
WK_SWIFT_UI_ACTOR NS_SWIFT_NAME(WKWebExtension.WindowConfiguration)
@interface WKWebExtensionWindowConfiguration : NSObject

+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;

/*! @abstract Indicates the window type for the window. */
@property (nonatomic, readonly) WKWebExtensionWindowType windowType;

/*! @abstract Indicates the window state for the window. */
@property (nonatomic, readonly) WKWebExtensionWindowState windowState;

/*!
 @abstract Indicates the frame where the window should be positioned on the main screen.
 @discussion This frame should override the app's default window position and size.
 Individual components (e.g., `origin.x`, `size.width`) will be `NaN` if not specified.
 */
@property (nonatomic, readonly) CGRect frame;

/*!
 @abstract Indicates the URLs that the window should initially load as tabs.
 @discussion If ``tabURLs`` and ``tabs`` are both empty, the app's default "start page" should appear in a tab.
 @seealso tabs
 */
@property (nonatomic, readonly, copy) NSArray<NSURL *> *tabURLs;

/*!
 @abstract Indicates the existing tabs that should be moved to the window.
 @discussion If ``tabs`` and ``tabURLs`` are both empty, the app's default "start page" should appear in a tab.
 @seealso tabURLs
 */
@property (nonatomic, readonly, copy) NSArray<id <WKWebExtensionTab>> *tabs;

/*! @abstract Indicates whether the window should be focused. */
@property (nonatomic, readonly) BOOL shouldBeFocused;

/*!
 @abstract Indicates whether the window should be private.
 @note To ensure proper isolation between private and non-private data, web views associated with private data must use a
 different ``WKUserContentController``. Likewise, to be identified as a private web view and to ensure that cookies and other
 website data is not shared, private web views must be configured to use a non-persistent ``WKWebsiteDataStore``.
 */
@property (nonatomic, readonly) BOOL shouldBePrivate;

@end

WK_HEADER_AUDIT_END(nullability, sendability)
