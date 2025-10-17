/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 25, 2023.
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
#pragma once

#if PLATFORM(IOS_FAMILY)

#import <UIKit/UIKit.h>

namespace WebKit {
enum class TabDirection : bool { Next, Previous };
}

@class WKFormAccessoryView;

@protocol WKFormAccessoryViewDelegate <NSObject>

- (void)accessoryViewDone:(WKFormAccessoryView *)view;
- (void)accessoryView:(WKFormAccessoryView *)view tabInDirection:(WebKit::TabDirection)direction;
- (void)accessoryViewAutoFill:(WKFormAccessoryView *)view;

@end

@interface WKFormAccessoryView : UIInputView

- (instancetype)initWithInputAssistantItem:(UITextInputAssistantItem *)inputAssistantItem delegate:(id<WKFormAccessoryViewDelegate>)delegate;
- (void)showAutoFillButtonWithTitle:(NSString *)title;
- (void)hideAutoFillButton;
- (void)setNextPreviousItemsVisible:(BOOL)visible;

@property (nonatomic, readonly) UIBarButtonItem *autoFillButtonItem;
@property (nonatomic, getter=isNextEnabled) BOOL nextEnabled;
@property (nonatomic, getter=isPreviousEnabled) BOOL previousEnabled;

@end

#endif // PLATFORM(IOS_FAMILY)
