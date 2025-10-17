/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 22, 2023.
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

#if HAVE(PEPPER_UI_CORE)

#import "WKQuickboardViewControllerDelegate.h"

typedef NS_ENUM(NSInteger, WKNumberPadInputMode) {
    WKNumberPadInputModeNone,
    WKNumberPadInputModeNumbersAndSymbols,
    WKNumberPadInputModeTelephone,
    WKNumberPadInputModeNumbersOnly
};

@class WKTextInputListViewController;

@protocol WKTextInputListViewControllerDelegate <WKQuickboardViewControllerDelegate>

- (WKNumberPadInputMode)numericInputModeForListViewController:(WKTextInputListViewController *)controller;
- (PUICTextInputContext *)textInputContextForListViewController:(WKTextInputListViewController *)controller;
- (UIView *)inputContextViewForViewController:(PUICQuickboardViewController *)controller;
- (BOOL)allowsDictationInputForListViewController:(WKTextInputListViewController *)controller;
- (BOOL)allowsLanguageSelectionForListViewController:(WKTextInputListViewController *)controller;
- (BOOL)shouldDisplayInputContextViewForListViewController:(PUICQuickboardViewController *)controller;

@end

@interface WKTextInputListViewController : PUICQuickboardMessageViewController

- (instancetype)initWithDelegate:(id <WKTextInputListViewControllerDelegate>)delegate NS_DESIGNATED_INITIALIZER;
- (instancetype)initWithDelegate:(id <PUICQuickboardViewControllerDelegate>)delegate dictationMode:(PUICDictationMode)dictationMode NS_UNAVAILABLE;
- (instancetype)initWithCoder:(NSCoder *)coder NS_UNAVAILABLE;
- (void)updateTextSuggestions:(NSArray<UITextSuggestion *> *)suggestions;
- (void)reloadContextView;

@property (nonatomic, weak) id <WKTextInputListViewControllerDelegate> delegate;

@end

@interface WKTextInputListViewController (Testing)

- (void)enterText:(NSString *)text;

@end

#endif // HAVE(PEPPER_UI_CORE)
