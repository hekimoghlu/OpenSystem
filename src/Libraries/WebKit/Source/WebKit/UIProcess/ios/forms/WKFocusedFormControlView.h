/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 17, 2023.
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
#if HAVE(PEPPER_UI_CORE)

#import "UIKitSPI.h"

@class WKFocusedFormControlView;

@protocol WKFocusedFormControlViewDelegate <NSObject>

- (void)focusedFormControlViewDidSubmit:(WKFocusedFormControlView *)view;
- (void)focusedFormControlViewDidCancel:(WKFocusedFormControlView *)view;
- (void)focusedFormControlViewDidBeginEditing:(WKFocusedFormControlView *)view;
- (CGRect)rectForFocusedFormControlView:(WKFocusedFormControlView *)view;
- (NSString *)actionNameForFocusedFormControlView:(WKFocusedFormControlView *)view;

// Support for focusing upstream and downstream nodes.
- (void)focusedFormControlViewDidRequestNextNode:(WKFocusedFormControlView *)view;
- (void)focusedFormControlViewDidRequestPreviousNode:(WKFocusedFormControlView *)view;
- (BOOL)hasNextNodeForFocusedFormControlView:(WKFocusedFormControlView *)view;
- (BOOL)hasPreviousNodeForFocusedFormControlView:(WKFocusedFormControlView *)view;
- (CGRect)nextRectForFocusedFormControlView:(WKFocusedFormControlView *)view;
- (CGRect)previousRectForFocusedFormControlView:(WKFocusedFormControlView *)view;
- (UIScrollView *)scrollViewForFocusedFormControlView:(WKFocusedFormControlView *)view;

- (void)focusedFormControllerDidUpdateSuggestions:(WKFocusedFormControlView *)view;
@end

@interface WKFocusedFormControlView : UIView <UITextInputSuggestionDelegate>

- (instancetype)initWithFrame:(CGRect)frame delegate:(id <WKFocusedFormControlViewDelegate>)delegate;
- (instancetype)initWithCoder:(NSCoder *)aDecoder NS_UNAVAILABLE;
- (instancetype)initWithFrame:(CGRect)frame NS_UNAVAILABLE;

- (void)reloadData:(BOOL)animated;
- (void)show:(BOOL)animated;
- (void)hide:(BOOL)animated;

- (void)engageFocusedFormControlNavigation;
- (void)disengageFocusedFormControlNavigation;

- (BOOL)handleWheelEvent:(UIEvent *)event;

@property (nonatomic, weak) id <WKFocusedFormControlViewDelegate> delegate;
@property (nonatomic, readonly, getter=isVisible) BOOL visible;
@property (nonatomic, copy) NSArray<UITextSuggestion *> *suggestions;

@end

#endif // HAVE(PEPPER_UI_CORE)
