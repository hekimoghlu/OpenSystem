/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 10, 2021.
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

@import Foundation;

@protocol UIApplicationDelegate <NSObject>
@end

@interface UIResponder : NSObject
@end

@interface UIView : UIResponder
@end

typedef NS_OPTIONS(NSUInteger, UIViewAnimationOptions) {
    UIViewAnimationOptionLayoutSubviews            = 1 <<  0,
    UIViewAnimationOptionTransitionFlipFromBottom  = 7 << 20,
  };
  
@interface UIView(UIViewAnimationWithBlocks)

+ (void)animateWithDuration:(NSTimeInterval)duration 
                      delay:(NSTimeInterval)delay 
                    options:(UIViewAnimationOptions)options 
                 animations:(void (^)(void))animations 
                 completion:(void (^)(BOOL finished))completion;

@end

@interface UIActionSheet : UIView

- (instancetype)initWithTitle:(NSString *)title
                     delegate:(id)delegate
            cancelButtonTitle:(NSString *)cancelButtonTitle
       destructiveButtonTitle:(NSString *)destructiveButtonTitle
            otherButtonTitles:(NSString *)titles, ...;

@end

@interface UIAlertView : UIView

- (instancetype)initWithTitle:(NSString *)title
                      message:(NSString *)message
                     delegate:(id)delegate
            cancelButtonTitle:(NSString *)cancelButtonTitle
            otherButtonTitles:(NSString *)titles, ...;

@end
