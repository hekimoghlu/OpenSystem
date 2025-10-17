/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 17, 2024.
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
#import <WebKit/WKFoundation.h>
#import <WebKit/_WKInspectorExtensionHost.h>
#import <WebKit/_WKInspectorIBActions.h>

#if !TARGET_OS_IPHONE

@class WKWebView;
@class _WKInspectorConfiguration;
@class _WKInspectorDebuggableInfo;

@protocol _WKRemoteWebInspectorViewControllerDelegate;

NS_ASSUME_NONNULL_BEGIN

WK_CLASS_AVAILABLE(macos(10.12.4))
@interface _WKRemoteWebInspectorViewController : NSObject <_WKInspectorExtensionHost, _WKInspectorIBActions>

@property (nonatomic, weak) id <_WKRemoteWebInspectorViewControllerDelegate> delegate;

@property (nonatomic, nullable, readonly, retain) NSWindow *window;
@property (nonatomic, nullable, readonly, retain) WKWebView *webView;
@property (nonatomic, readonly, copy) _WKInspectorConfiguration *configuration WK_API_AVAILABLE(macos(12.0));

- (instancetype)initWithConfiguration:(_WKInspectorConfiguration *)configuration WK_API_AVAILABLE(macos(12.0));
- (void)loadForDebuggable:(_WKInspectorDebuggableInfo *)debuggableInfo backendCommandsURL:(NSURL *)backendCommandsURL WK_API_AVAILABLE(macos(12.0));

- (void)sendMessageToFrontend:(NSString *)message;

@end

@protocol _WKRemoteWebInspectorViewControllerDelegate <NSObject>
@optional
- (void)inspectorViewController:(_WKRemoteWebInspectorViewController *)controller sendMessageToBackend:(NSString *)message;
- (void)inspectorViewControllerInspectorDidClose:(_WKRemoteWebInspectorViewController *)controller;
@end

NS_ASSUME_NONNULL_END

#endif // !TARGET_OS_IPHONE
