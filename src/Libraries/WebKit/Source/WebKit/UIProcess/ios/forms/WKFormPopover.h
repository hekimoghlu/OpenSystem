/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 16, 2025.
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

#import <UIKit/UIPopoverController.h>

@class WKContentView;
@protocol WKRotatingPopoverDelegate;

@interface WKRotatingPopover : NSObject

- (id)initWithView:(WKContentView *)view;
- (void)presentPopoverAnimated:(BOOL)animated;
- (void)dismissPopoverAnimated:(BOOL)animated;
- (UIPopoverArrowDirection)popoverArrowDirections;

@property (nonatomic, readonly) WKContentView *view;

@property (nonatomic, assign) CGPoint presentationPoint;
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
@property (nonatomic, retain) UIPopoverController *popoverController;
ALLOW_DEPRECATED_DECLARATIONS_END
@property (nonatomic, assign) id <WKRotatingPopoverDelegate> dismissionDelegate;
@end


@protocol WKRotatingPopoverDelegate
- (void)popoverWasDismissed:(WKRotatingPopover *)popover;
@end

@interface WKFormRotatingAccessoryPopover : WKRotatingPopover <WKRotatingPopoverDelegate>
- (void)accessoryDone;
@end

#endif // PLATFORM(IOS_FAMILY)
