/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 19, 2023.
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

#import <WebKit/WKFoundation.h>

#if TARGET_OS_OSX

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@class _WKFrameHandle;
@class _WKInspectorExtension;

@protocol _WKInspectorExtensionDelegate <NSObject>
@optional

/**
 * @abstract Called when a tab associated with this extension has been shown.
 * @param extension The extension that created the shown tab.
 * @param tabIdentifier Identifier for the tab that was shown.
 */
- (void)inspectorExtension:(_WKInspectorExtension *)extension didShowTabWithIdentifier:(NSString *)tabIdentifier withFrameHandle:(_WKFrameHandle *)frameHandle;

/**
 * @abstract Called when a tab associated with this extension has been hidden.
 * @param extension The extension that created the hidden tab.
 * @param tabIdentifier Identifier for the tab that was hidden.
 */
- (void)inspectorExtension:(_WKInspectorExtension *)extension didHideTabWithIdentifier:(NSString *)tabIdentifier;

/**
 * @abstract Called when a tab associated with this extension has navigated to a new URL.
 * @param extension The extension that created the tab.
 * @param tabIdentifier Identifier for the tab that navigated.
 * @param URL The new URL for the extension tab's page.
 */
- (void)inspectorExtension:(_WKInspectorExtension *)extension didNavigateTabWithIdentifier:(NSString *)tabIdentifier newURL:(NSURL *)newURL;

/**
 * @abstract Called when the inspected page has navigated to a new URL.
 * @param extension The extension that is being notified.
 * @param url The new URL for the inspected page.
 */
- (void)inspectorExtension:(_WKInspectorExtension *)extension inspectedPageDidNavigate:(NSURL *)url;

@end

NS_ASSUME_NONNULL_END

#endif // TARGET_OS_OSX
