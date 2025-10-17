/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 24, 2024.
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
#import "MBCBoardMTLView.h"

NS_ASSUME_NONNULL_BEGIN

/*!
 @abstract MBCBoardMTLAccessibilityProxy is used to provide accessibility features for each square on the board.
 */
@interface MBCBoardMTLAccessibilityProxy : NSObject {
    MBCBoardMTLView *_view;
    MBCSquare _square;
}

/*!
 @abstract proxyWithView:square:
 @param view The main view for rendering
 @param square The square that the MBCBoardMTLAccessibilityProxy represents
 @discussion Returns an instance of the MBCBoardMTLAccessibilityProxy
 */
+ (id)proxyWithView:(MBCBoardMTLView *)view square:(MBCSquare)square;

/*!
 @abstract initWithView:square:
 @param view The main view for rendering
 @param square The square that the MBCBoardMTLAccessibilityProxy represents
 @discussion The default initializer
 */
- (id)initWithView:(MBCBoardMTLView *)view square:(MBCSquare)square;

@end

/*!
 @abstract A category to provide accessibility features for the main view
 */
@interface MBCBoardMTLView (Accessibility)

/*!
 @abstract describeSquare:
 @param sqaure The square to be described as a string
 @discussion Will return a string that describes the type and color of piece as well as position on board.
 */
- (NSString *)describeSquare:(MBCSquare)square;

/*!
 @abstract selectSquare:
 @param square The square to select
 @discussion Will be called if accessibilityPerformAction is called with NSAccessibilityPressAction on
 the MBCBoardMTLAccessibilityProxy associated with a square
 */
- (void)selectSquare:(MBCSquare)square;

@end

NS_ASSUME_NONNULL_END
