/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 3, 2023.
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
#import <WebKit/_WKInspectorExtensionHost.h>
#import <WebKit/_WKInspectorIBActions.h>

NS_ASSUME_NONNULL_BEGIN

@class WKWebView;
@class _WKFrameHandle;
@class _WKInspectorExtension;
@protocol _WKInspectorDelegate;

WK_CLASS_AVAILABLE(macos(10.14.4), ios(12.2))
@interface _WKInspector : NSObject <_WKInspectorExtensionHost, _WKInspectorIBActions>

- (instancetype)init NS_UNAVAILABLE;

@property (nonatomic, weak) id <_WKInspectorDelegate> delegate WK_API_AVAILABLE(macos(12.0), ios(15.0));

@property (nonatomic, readonly) WKWebView *webView;
@property (nonatomic, readonly) BOOL isConnected;
@property (nonatomic, readonly) BOOL isVisible;
@property (nonatomic, readonly) BOOL isFront;
@property (nonatomic, readonly) BOOL isProfilingPage;
@property (nonatomic, readonly) BOOL isElementSelectionActive;

- (void)connect;
- (void)hide;
- (void)showMainResourceForFrame:(_WKFrameHandle *)frame;
- (void)attach;
- (void)detach;
- (void)togglePageProfiling;
- (void)toggleElementSelection;
- (void)printErrorToConsole:(NSString *)error;

@end

NS_ASSUME_NONNULL_END
