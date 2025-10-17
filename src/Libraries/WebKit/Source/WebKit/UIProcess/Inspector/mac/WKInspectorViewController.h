/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 6, 2024.
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
#if PLATFORM(MAC)

#import <WebKit/WKFoundation.h>
#import <wtf/NakedPtr.h>

OBJC_CLASS WKWebView;
OBJC_CLASS _WKInspectorConfiguration;

namespace WebKit {
class WebPageProxy;
}

@protocol WKInspectorViewControllerDelegate;

NS_ASSUME_NONNULL_BEGIN

@interface WKInspectorViewController : NSObject

@property (nonatomic, readonly) WKWebView *webView;
@property (nonatomic, weak) id <WKInspectorViewControllerDelegate> delegate;

- (instancetype)initWithConfiguration:(_WKInspectorConfiguration *)configuration inspectedPage:(NakedPtr<WebKit::WebPageProxy>)inspectedPage;

+ (BOOL)viewIsInspectorWebView:(NSView *)view;
+ (NSURL * _Nullable)URLForInspectorResource:(NSString *)resource;

@end

@protocol WKInspectorViewControllerDelegate <NSObject>
@optional
- (void)inspectorViewControllerDidBecomeActive:(WKInspectorViewController *)inspectorViewController;
- (void)inspectorViewControllerInspectorDidCrash:(WKInspectorViewController *)inspectorViewController;
- (BOOL)inspectorViewControllerInspectorIsUnderTest:(WKInspectorViewController *)inspectorViewController;
- (void)inspectorViewController:(WKInspectorViewController *)inspectorViewController willMoveToWindow:(NSWindow *)newWindow;
- (void)inspectorViewControllerDidMoveToWindow:(WKInspectorViewController *)inspectorViewController;
- (void)inspectorViewController:(WKInspectorViewController *)inspectorViewController openURLExternally:(NSURL *)url;
@end

NS_ASSUME_NONNULL_END

#endif // PLATFORM(MAC)
