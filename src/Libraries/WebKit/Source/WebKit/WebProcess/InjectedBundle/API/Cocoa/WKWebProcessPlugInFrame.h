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

#import <CoreGraphics/CoreGraphics.h>
#import <Foundation/Foundation.h>
#import <JavaScriptCore/JavaScriptCore.h>

@class _WKFrameHandle;
@class WKWebProcessPlugInCSSStyleDeclarationHandle;
@class WKWebProcessPlugInHitTestResult;
@class WKWebProcessPlugInNodeHandle;
@class WKWebProcessPlugInRangeHandle;
@class WKWebProcessPlugInScriptWorld;

typedef NS_OPTIONS(NSUInteger, WKHitTestOptions) {
    WKHitTestOptionAllowUserAgentShadowRootContent = 1 << 0,
} WK_API_AVAILABLE(macos(12.0), ios(15.0));

WK_CLASS_AVAILABLE(macos(10.10), ios(8.0))
@interface WKWebProcessPlugInFrame : NSObject

@property (nonatomic, readonly) NSURL *URL;
@property (nonatomic, readonly) NSArray *childFrames WK_API_DEPRECATED("Child frames might not be in the same process", macos(10.10, 14.4), ios(8.0, 17.4), visionos(1.0, 1.1));
@property (nonatomic, readonly) BOOL containsAnyFormElements;
@property (nonatomic, readonly) BOOL isMainFrame;

@property (nonatomic, readonly) _WKFrameHandle *handle;

// Returns URLs ordered by resolution in descending order.
// FIXME: These should be tagged nonnull.
@property (nonatomic, readonly) NSArray<NSURL *> *appleTouchIconURLs WK_API_AVAILABLE(macos(10.12), ios(10.0));
@property (nonatomic, readonly) NSArray<NSURL *> *faviconURLs WK_API_AVAILABLE(macos(10.12), ios(10.0));

- (JSContext *)jsContextForWorld:(WKWebProcessPlugInScriptWorld *)world;
- (JSContext *)jsContextForServiceWorkerWorld:(WKWebProcessPlugInScriptWorld *)world;
- (WKWebProcessPlugInHitTestResult *)hitTest:(CGPoint)point;
- (WKWebProcessPlugInHitTestResult *)hitTest:(CGPoint)point options:(WKHitTestOptions)options WK_API_AVAILABLE(macos(12.0), ios(15.0));
- (JSValue *)jsCSSStyleDeclarationForCSSStyleDeclarationHandle:(WKWebProcessPlugInCSSStyleDeclarationHandle *)cssStyleDeclarationHandle inWorld:(WKWebProcessPlugInScriptWorld *)world WK_API_AVAILABLE(macos(13.0), ios(16.0));
- (JSValue *)jsNodeForNodeHandle:(WKWebProcessPlugInNodeHandle *)nodeHandle inWorld:(WKWebProcessPlugInScriptWorld *)world;
- (JSValue *)jsRangeForRangeHandle:(WKWebProcessPlugInRangeHandle *)rangeHandle inWorld:(WKWebProcessPlugInScriptWorld *)world WK_API_AVAILABLE(macos(10.12.4), ios(10.3));

@end
