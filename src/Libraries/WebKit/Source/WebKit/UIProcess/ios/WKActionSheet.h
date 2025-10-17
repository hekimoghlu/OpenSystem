/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 25, 2024.
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
#if PLATFORM(IOS_FAMILY)

#import <UIKit/UIAlertController.h>
#import <UIKit/UIPopoverController.h>

typedef NS_ENUM(NSInteger, WKActionSheetPresentationStyle) {
    WKActionSheetPresentAtTouchLocation,
    WKActionSheetPresentAtElementRect,
    WKActionSheetPresentAtClosestIndicatorRect
};

@protocol WKActionSheetDelegate;
@class WKContentView;

@interface WKActionSheet : UIAlertController

@property (nonatomic, assign) id <WKActionSheetDelegate> sheetDelegate;
@property (nonatomic) UIPopoverArrowDirection arrowDirections;
- (void)doneWithSheet:(BOOL)dismiss;
- (BOOL)presentSheet:(WKActionSheetPresentationStyle)style;
- (BOOL)presentSheetFromRect:(CGRect)presentationRect;
- (void)updateSheetPosition;
@end

@protocol WKActionSheetDelegate
@required
- (UIView *)hostViewForSheet;
- (CGRect)initialPresentationRectInHostViewForSheet;
- (CGRect)presentationRectForIndicatedElement;
- (CGRect)presentationRectForElementUsingClosestIndicatedRect;
- (CGRect)presentationRectInHostViewForSheet;
- (void)updatePositionInformation;
@end

#endif // PLATFORM(IOS_FAMILY)
