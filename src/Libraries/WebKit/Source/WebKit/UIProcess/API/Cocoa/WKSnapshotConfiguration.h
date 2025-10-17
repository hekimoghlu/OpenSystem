/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 10, 2022.
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

#import <CoreGraphics/CGGeometry.h>
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

WK_SWIFT_UI_ACTOR
WK_CLASS_AVAILABLE(macos(10.13), ios(11.0))
@interface WKSnapshotConfiguration : NSObject <NSCopying>

/*! @abstract The rect to snapshot in view coordinates.
 @discussion This rect should be contained within WKWebView's bounds. If the rect is set to the 
 null rect, the view's bounds will be used. The initial value is the null rect.
 */
@property (nonatomic) CGRect rect;

/*! @abstract Specify a custom width to control the size of image you get back. The height will be 
 computed to maintain the aspect ratio established by rect.
 @discussion snapshotWidth represents the width in points. If the snapshotWidth is nil, rect's
 width will be used.
 */
@property (nullable, nonatomic, copy) NSNumber *snapshotWidth;

/*! @abstract A Boolean value that specifies whether the snapshot should be taken after recent
 changes have been incorporated. The value NO will capture the screen in its current state,
 which might not include recent changes.
 @discussion The default value is YES.
 */
@property (nonatomic) BOOL afterScreenUpdates WK_API_AVAILABLE(macos(10.15), ios(13.0));

@end

NS_ASSUME_NONNULL_END
