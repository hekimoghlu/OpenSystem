/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 1, 2023.
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

#if PLATFORM(IOS_FAMILY)

#import <WebKit/WKPageLoadTypes.h>
#import <WebKit/_WKFindOptions.h>

@class NSData;
@class UIEvent;
@class UIScrollView;
@class UIView;
@class WKWebView;
@protocol NSObject;
struct CGSize;

#ifdef FOUNDATION_HAS_DIRECTIONAL_GEOMETRY
typedef NSEdgeInsets UIEdgeInsets;
#else
struct UIEdgeInsets;
#endif

@protocol WKWebViewContentProvider <NSObject>

- (instancetype)web_initWithFrame:(CGRect)frame webView:(WKWebView *)webView mimeType:(NSString *)mimeType __attribute__((objc_method_family(init)));
- (void)web_setContentProviderData:(NSData *)data suggestedFilename:(NSString *)filename;
- (void)web_setMinimumSize:(CGSize)size;
- (void)web_setOverlaidAccessoryViewsInset:(CGSize)inset;
- (void)web_computedContentInsetDidChange;
- (void)web_setFixedOverlayView:(UIView *)fixedOverlayView;
- (void)web_didSameDocumentNavigation:(WKSameDocumentNavigationType)navigationType;
- (void)web_countStringMatches:(NSString *)string options:(_WKFindOptions)options maxCount:(NSUInteger)maxCount;
- (void)web_findString:(NSString *)string options:(_WKFindOptions)options maxCount:(NSUInteger)maxCount;
- (void)web_hideFindUI;
@property (nonatomic, readonly) UIView *web_contentView;
@property (nonatomic, readonly, class) BOOL web_requiresCustomSnapshotting;

@optional
- (void)web_scrollViewDidScroll:(UIScrollView *)scrollView;
- (void)web_scrollViewWillBeginZooming:(UIScrollView *)scrollView withView:(UIView *)view;
- (void)web_scrollViewDidEndZooming:(UIScrollView *)scrollView withView:(UIView *)view atScale:(CGFloat)scale;
- (void)web_scrollViewDidZoom:(UIScrollView *)scrollView;
- (void)web_beginAnimatedResizeWithUpdates:(void (^)(void))updateBlock;
- (BOOL)web_handleKeyEvent:(UIEvent *)event;
- (void)web_snapshotRectInContentViewCoordinates:(CGRect)contentViewCoordinates snapshotWidth:(CGFloat)snapshotWidth completionHandler:(void (^)(CGImageRef))completionHandler;
@property (nonatomic, readonly) NSData *web_dataRepresentation;
@property (nonatomic, readonly) NSString *web_suggestedFilename;
@property (nonatomic, readonly) BOOL web_isBackground;

@end

#endif // PLATFORM(IOS_FAMILY)

