/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 30, 2023.
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
#import "config.h"
#import "WKTextInputListViewController.h"

#if HAVE(PEPPER_UI_CORE)

#import "WKNumberPadViewController.h"
#import <wtf/RetainPtr.h>

@implementation WKTextInputListViewController {
    BOOL _contextViewNeedsUpdate;
    RetainPtr<UIView> _contextView;
    RetainPtr<WKNumberPadViewController> _numberPadViewController;
}

@dynamic delegate;

- (instancetype)initWithDelegate:(id <WKTextInputListViewControllerDelegate>)delegate
{
    if (!(self = [super initWithDelegate:delegate dictationMode:PUICDictationModeText]))
        return nil;

    _contextViewNeedsUpdate = YES;
    self.textInputContext = [self.delegate textInputContextForListViewController:self];
    return self;
}

- (void)viewDidLoad
{
    [super viewDidLoad];

    self.view.backgroundColor = UIColor.systemBackgroundColor;
}

- (void)reloadContextView
{
    _contextViewNeedsUpdate = YES;
    [self reloadHeaderContentView];
}

- (void)updateContextViewIfNeeded
{
    if (!_contextViewNeedsUpdate)
        return;

    auto previousContextView = _contextView;
    if ([self.delegate shouldDisplayInputContextViewForListViewController:self])
        _contextView = [self.delegate inputContextViewForViewController:self];
    else
        _contextView = nil;

    _contextViewNeedsUpdate = NO;
}

- (BOOL)requiresNumericInput
{
    return [self.delegate numericInputModeForListViewController:self] != WKNumberPadInputModeNone;
}

- (NSArray *)additionalTrayButtons
{
    if (!self.requiresNumericInput)
        return @[ ];

#if HAVE(PUIC_BUTTON_TYPE_PILL)
    auto numberPadButton = retainPtr([PUICQuickboardListTrayButton buttonWithType:PUICButtonTypePill]);
#else
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    auto numberPadButton = adoptNS([[PUICQuickboardListTrayButton alloc] initWithFrame:CGRectZero tintColor:nil defaultHeight:self.specs.defaultButtonHeight]);
ALLOW_DEPRECATED_DECLARATIONS_END
#endif
    [numberPadButton setAction:PUICQuickboardActionAddNumber];
    [numberPadButton addTarget:self action:@selector(presentNumberPadViewController) forControlEvents:UIControlEventTouchUpInside];
    return @[ numberPadButton.get() ];
}

- (void)presentNumberPadViewController
{
    if (_numberPadViewController)
        return;

    WKNumberPadInputMode mode = [self.delegate numericInputModeForListViewController:self];
    if (mode == WKNumberPadInputModeNone) {
        ASSERT_NOT_REACHED();
        return;
    }

    NSString *initialText = [self.delegate initialValueForViewController:self];
    _numberPadViewController = adoptNS([[WKNumberPadViewController alloc] initWithDelegate:self.delegate initialText:initialText inputMode:mode]);
    [self presentViewController:_numberPadViewController.get() animated:YES completion:nil];
}

- (void)updateTextSuggestions:(NSArray<UITextSuggestion *> *)suggestions
{
    auto messages = adoptNS([[NSMutableArray<NSAttributedString *> alloc] initWithCapacity:suggestions.count]);
    for (UITextSuggestion *suggestion in suggestions) {
        auto attributedString = adoptNS([[NSAttributedString alloc] initWithString:suggestion.displayText]);
        [messages addObject:attributedString.get()];
    }
    self.messages = messages.get();
}

- (void)enterText:(NSString *)text
{
    ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    [self.delegate quickboard:static_cast<id<PUICQuickboardController>>(self) textEntered:adoptNS([[NSAttributedString alloc] initWithString:text]).get()];
    ALLOW_DEPRECATED_DECLARATIONS_END
}

#pragma mark - Quickboard subclassing

- (CGFloat)headerContentViewHeight
{
    [self updateContextViewIfNeeded];

    return [_contextView sizeThatFits:self.contentView.bounds.size].height;
}

- (UIView *)headerContentView
{
    [self updateContextViewIfNeeded];

    CGFloat viewWidth = CGRectGetWidth(self.contentView.bounds);
    CGSize sizeThatFits = [_contextView sizeThatFits:self.contentView.bounds.size];
    [_contextView setFrame:CGRectMake((viewWidth - sizeThatFits.width) / 2, 0, sizeThatFits.width, sizeThatFits.height)];
    return _contextView.get();
}

- (BOOL)shouldShowLanguageButton
{
    return [self.delegate allowsLanguageSelectionForListViewController:self];
}

- (BOOL)supportsDictationInput
{
    return [self.delegate allowsDictationInputForListViewController:self];
}

- (BOOL)shouldShowTrayView
{
    return self.requiresNumericInput;
}

- (BOOL)shouldShowTextField
{
    return !self.requiresNumericInput;
}

- (BOOL)supportsArouetInput
{
    return !self.requiresNumericInput;
}

@end

#endif // HAVE(PEPPER_UI_CORE)
