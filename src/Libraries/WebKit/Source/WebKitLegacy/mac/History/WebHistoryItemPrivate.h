/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 25, 2025.
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
#import <CoreGraphics/CoreGraphics.h>
#import <WebKitLegacy/WebHistoryItem.h>

#if TARGET_OS_IPHONE
extern NSString *WebViewportInitialScaleKey;
extern NSString *WebViewportMinimumScaleKey;
extern NSString *WebViewportMaximumScaleKey;
extern NSString *WebViewportUserScalableKey;
extern NSString *WebViewportShrinkToFitKey;
extern NSString *WebViewportFitKey;
extern NSString *WebViewportWidthKey;
extern NSString *WebViewportHeightKey;

extern NSString *WebViewportFitAutoValue;
extern NSString *WebViewportFitContainValue;
extern NSString *WebViewportFitCoverValue;
#endif

@interface WebHistoryItem (WebPrivate)

#if !TARGET_OS_IPHONE
+ (void)_releaseAllPendingPageCaches;
#endif

- (id)initWithURL:(NSURL *)URL title:(NSString *)title;

- (NSURL *)URL;
- (BOOL)lastVisitWasFailure;

- (NSString *)RSSFeedReferrer;
- (void)setRSSFeedReferrer:(NSString *)referrer;

- (NSArray *)_redirectURLs;

- (NSString *)target;
- (BOOL)isTargetItem;
- (NSArray *)children;
- (NSDictionary *)dictionaryRepresentation;
#if TARGET_OS_IPHONE
- (NSDictionary *)dictionaryRepresentationIncludingChildren:(BOOL)includesChildren;
#endif

#if TARGET_OS_IPHONE
- (void)_setScale:(float)scale isInitial:(BOOL)aFlag;
- (float)_scale;
- (BOOL)_scaleIsInitial;
- (NSDictionary *)_viewportArguments;
- (void)_setViewportArguments:(NSDictionary *)arguments;
- (CGPoint)_scrollPoint;
- (void)_setScrollPoint:(CGPoint)scrollPoint;
#endif

- (BOOL)_isInBackForwardCache;
- (BOOL)_hasCachedPageExpired;

@end
