/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 19, 2023.
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
#import <WebKit/_WKRectEdge.h>

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSInteger, _WKTargetedElementPosition) {
    _WKTargetedElementPositionStatic,
    _WKTargetedElementPositionRelative,
    _WKTargetedElementPositionAbsolute,
    _WKTargetedElementPositionSticky,
    _WKTargetedElementPositionFixed
} WK_API_AVAILABLE(macos(15.0), ios(18.0), visionos(2.0));

@class _WKFrameTreeNode;

WK_CLASS_AVAILABLE(macos(15.0), ios(18.0), visionos(2.0))
@interface _WKTargetedElementInfo : NSObject

@property (nonatomic, readonly) _WKTargetedElementPosition positionType;
@property (nonatomic, readonly) CGRect boundsInWebView; // In WKWebView's coordinate space.
@property (nonatomic, readonly) CGRect boundsInClientCoordinates;
@property (nonatomic, readonly, getter=isNearbyTarget) BOOL nearbyTarget;
@property (nonatomic, readonly, getter=isPseudoElement) BOOL pseudoElement;
@property (nonatomic, readonly, getter=isInShadowTree) BOOL inShadowTree;
@property (nonatomic, readonly, getter=isInVisibilityAdjustmentSubtree) BOOL inVisibilityAdjustmentSubtree;
@property (nonatomic, readonly) BOOL hasLargeReplacedDescendant;
@property (nonatomic, readonly) BOOL hasAudibleMedia;
@property (nonatomic, readonly) NSSet<NSURL *> *mediaAndLinkURLs;

@property (nonatomic, readonly, copy) NSArray<NSString *> *selectors;
@property (nonatomic, readonly, copy) NSArray<NSArray<NSString *> *> *selectorsIncludingShadowHosts;
@property (nonatomic, readonly, copy) NSString *renderedText;
@property (nonatomic, readonly, copy) NSString *searchableText;
@property (nonatomic, readonly, copy) NSString *screenReaderText;
@property (nonatomic, readonly) _WKRectEdge offsetEdges;

// In root view coordinates. To be deprecated and removed, once clients adopt the more explicit bounds properties above.
@property (nonatomic, readonly) CGRect bounds;

- (BOOL)isSameElement:(_WKTargetedElementInfo *)other;

- (void)getChildFrames:(void(^)(NSArray<_WKFrameTreeNode *> *))completionHandler;
- (void)takeSnapshotWithCompletionHandler:(void(^)(CGImageRef))completionHandler;

@end

NS_ASSUME_NONNULL_END
