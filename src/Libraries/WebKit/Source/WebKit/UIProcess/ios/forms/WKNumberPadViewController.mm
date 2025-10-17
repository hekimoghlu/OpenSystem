/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 18, 2022.
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
#import "WKNumberPadViewController.h"

#if HAVE(PEPPER_UI_CORE)

#import "UIKitSPI.h"
#import "WKNumberPadView.h"
#import <WebCore/LocalizedStrings.h>
#import <wtf/RetainPtr.h>
#import <wtf/WeakObjCPtr.h>
#import <wtf/text/WTFString.h>

static const CGFloat numberPadViewTopMargin = 30;
static const CGFloat headerButtonWidth = 24;
static const CGFloat inputLabelMinimumScale = 0.7;
static const CGFloat numberPadViewDismissAnimationDuration = 0.3;
static const NSTimeInterval numberPadDeleteKeyRepeatDelay = 0.35;
static const NSTimeInterval numberPadDeleteKeyRepeatInterval = 0.1;
static CGFloat inputLabelFontSize()
{
    if ([[UIDevice currentDevice] puic_deviceVariant] == PUICDeviceVariantCompact)
        return 16;
    return 18;
}

@implementation WKNumberPadViewController {
    RetainPtr<NSMutableString> _inputText;
    RetainPtr<WKNumberPadView> _numberPadView;
    WKNumberPadInputMode _inputMode;
    RetainPtr<UILabel> _inputLabel;
    RetainPtr<UIButton> _deleteButton;
    RetainPtr<UIButton> _backChevronButton;
    BOOL _shouldDismissWithFadeAnimation;
}

- (instancetype)initWithDelegate:(id <PUICQuickboardViewControllerDelegate>)delegate initialText:(NSString *)initialText inputMode:(WKNumberPadInputMode)inputMode
{
    if (!(self = [super initWithDelegate:delegate]))
        return nil;

    _inputText = adoptNS(initialText.mutableCopy);
    _inputMode = inputMode;
    _shouldDismissWithFadeAnimation = NO;
    return self;
}

- (void)dealloc
{
    [NSObject cancelPreviousPerformRequestsWithTarget:self];
    [super dealloc];
}

- (void)viewDidLoad
{
    [super viewDidLoad];

    self.view.backgroundColor = UIColor.systemBackgroundColor;

    _numberPadView = adoptNS([[WKNumberPadView alloc] initWithFrame:UIRectInset(self.contentView.bounds, numberPadViewTopMargin, 0, 0, 0) controller:self]);
    [self.contentView addSubview:_numberPadView.get()];

    _inputLabel = adoptNS([[UILabel alloc] init]);
    [_inputLabel setFont:[UIFont systemFontOfSize:inputLabelFontSize() weight:UIFontWeightSemibold]];
    [_inputLabel setTextColor:[UIColor whiteColor]];
    [_inputLabel setLineBreakMode:NSLineBreakByTruncatingHead];
    [_inputLabel setTextAlignment:NSTextAlignmentCenter];
    [_inputLabel setMinimumScaleFactor:inputLabelMinimumScale];
    [_inputLabel setAdjustsFontSizeToFitWidth:YES];
    [self.headerView addSubview:_inputLabel.get()];

    _deleteButton = [UIButton buttonWithType:UIButtonTypeCustom];
    UIImage *deleteButtonIcon = [[PUICResources imageNamed:@"keypad-delete-glyph" inBundle:[NSBundle bundleWithIdentifier:@"com.apple.PepperUICore"] shouldCache:YES] imageWithRenderingMode:UIImageRenderingModeAlwaysTemplate];
    [_deleteButton setImage:deleteButtonIcon forState:UIControlStateNormal];
    [_deleteButton setTintColor:[UIColor systemRedColor]];
    [_deleteButton addTarget:self action:@selector(_startDeletionTimer) forControlEvents:UIControlEventTouchDown];
    [_deleteButton addTarget:self action:@selector(_deleteButtonPressed) forControlEvents:UIControlEventTouchUpInside];
    [_deleteButton addTarget:self action:@selector(_cancelDeletionTimers) forControlEvents:UIControlEventTouchUpOutside];
    [self.headerView addSubview:_deleteButton.get()];

    _backChevronButton = [UIButton buttonWithType:UIButtonTypeCustom];
    UIImage *backChevronButtonIcon = [[PUICResources imageNamed:@"status-bar-chevron" inBundle:[NSBundle bundleWithIdentifier:@"com.apple.PepperUICore"] shouldCache:YES] imageWithRenderingMode:UIImageRenderingModeAlwaysTemplate];
    [_backChevronButton setImage:backChevronButtonIcon forState:UIControlStateNormal];
    [_backChevronButton setTintColor:[UIColor systemGrayColor]];
    [_backChevronButton addTarget:self action:@selector(_cancelInput) forControlEvents:UIControlEventTouchUpInside];
    [self.headerView addSubview:_backChevronButton.get()];

    [self _reloadHeaderViewFromInputText];
}

- (void)viewWillDisappear:(BOOL)animated
{
    [super viewWillDisappear:animated];
    [self _cancelDeletionTimers];
}

- (void)viewWillLayoutSubviews
{
    [super viewWillLayoutSubviews];

    [_inputLabel setFrame:UIRectInsetEdges(self.headerView.bounds, UIRectEdgeLeft | UIRectEdgeRight, headerButtonWidth)];
    [_deleteButton setFrame:CGRectMake(CGRectGetWidth(self.headerView.bounds) - headerButtonWidth, 0, headerButtonWidth, CGRectGetHeight(self.headerView.bounds))];
    [_backChevronButton setFrame:CGRectMake(0, 0, headerButtonWidth, CGRectGetHeight(self.headerView.bounds))];
}

- (void)_reloadHeaderViewFromInputText
{
    BOOL hasInputText = [_inputText length];
    self.cancelButton.hidden = hasInputText;
    [_deleteButton setHidden:!hasInputText];
    [_backChevronButton setHidden:!hasInputText];
    [_inputLabel setText:_inputText.get()];
}

- (void)didSelectKey:(WKNumberPadKey)key
{
    [self _handleKeyPress:key];
}

- (void)_handleKeyPress:(WKNumberPadKey)key
{
    switch (key) {
    case WKNumberPadKeyDash:
        [_inputText appendString:@"-"];
        break;
    case WKNumberPadKeyAsterisk:
        [_inputText appendString:@"*"];
        break;
    case WKNumberPadKeyOctothorpe:
        [_inputText appendString:@"#"];
        break;
    case WKNumberPadKeyClosingParenthesis:
        [_inputText appendString:@")"];
        break;
    case WKNumberPadKeyOpeningParenthesis:
        [_inputText appendString:@"("];
        break;
    case WKNumberPadKeyPlus:
        [_inputText appendString:@"+"];
        break;
    case WKNumberPadKeyAccept:
        ALLOW_DEPRECATED_DECLARATIONS_BEGIN
        [self.delegate quickboard:static_cast<id<PUICQuickboardController>>(self) textEntered:adoptNS([[NSAttributedString alloc] initWithString:_inputText.get()]).get()];
        ALLOW_DEPRECATED_DECLARATIONS_END
        return;
    case WKNumberPadKey0:
        [_inputText appendString:@"0"];
        break;
    case WKNumberPadKey1:
        [_inputText appendString:@"1"];
        break;
    case WKNumberPadKey2:
        [_inputText appendString:@"2"];
        break;
    case WKNumberPadKey3:
        [_inputText appendString:@"3"];
        break;
    case WKNumberPadKey4:
        [_inputText appendString:@"4"];
        break;
    case WKNumberPadKey5:
        [_inputText appendString:@"5"];
        break;
    case WKNumberPadKey6:
        [_inputText appendString:@"6"];
        break;
    case WKNumberPadKey7:
        [_inputText appendString:@"7"];
        break;
    case WKNumberPadKey8:
        [_inputText appendString:@"8"];
        break;
    case WKNumberPadKey9:
        [_inputText appendString:@"9"];
        break;
    default:
        break;
    }

    [self _cancelDeletionTimers];
    [self _reloadHeaderViewFromInputText];
}

- (void)_cancelInput
{
    _shouldDismissWithFadeAnimation = YES;
    ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    [self.delegate quickboardInputCancelled:static_cast<id<PUICQuickboardController>>(self)];
    ALLOW_DEPRECATED_DECLARATIONS_END
}

- (void)_deleteLastInputCharacter
{
    if (![_inputText length])
        return;

    [_inputText deleteCharactersInRange:NSMakeRange([_inputText length] - 1, 1)];
    [self _reloadHeaderViewFromInputText];
}

- (void)_deleteButtonPressed
{
    [self _deleteLastInputCharacter];
    [self _cancelDeletionTimers];
}

- (void)_cancelDeletionTimers
{
    [NSObject cancelPreviousPerformRequestsWithTarget:self selector:@selector(_startDeletionTimer) object:nil];
    [NSObject cancelPreviousPerformRequestsWithTarget:self selector:@selector(_deletionTimerFired) object:nil];
}

- (void)_startDeletionTimer
{
    [self _cancelDeletionTimers];
    [self performSelector:@selector(_deletionTimerFired) withObject:nil afterDelay:numberPadDeleteKeyRepeatDelay];
}

- (void)_deletionTimerFired
{
    [self _cancelDeletionTimers];
    [self _deleteLastInputCharacter];
    if ([_inputText length])
        [self performSelector:@selector(_deletionTimerFired) withObject:nil afterDelay:numberPadDeleteKeyRepeatInterval];
}

#pragma mark - PUICQuickboardViewController overrides

- (void)addContentViewAnimations:(BOOL)isPresenting
{
    if (!_shouldDismissWithFadeAnimation) {
        [super addContentViewAnimations:isPresenting];
        return;
    }

    CABasicAnimation *fadeOutAnimation = [CABasicAnimation animationWithKeyPath:@"opacity"];
    fadeOutAnimation.fromValue = @1;
    fadeOutAnimation.toValue = @0;
    fadeOutAnimation.duration = numberPadViewDismissAnimationDuration;
    fadeOutAnimation.timingFunction = [CAMediaTimingFunction functionWithName:kCAMediaTimingFunctionEaseInEaseOut];
#if USE(APPLE_INTERNAL_SDK)
    [self.contentView addAnimation:fadeOutAnimation forKey:@"WebKitNumberPadFadeOutAnimationKey"];
#endif
    self.contentView.alpha = 0;
}

@end

#endif // HAVE(PEPPER_UI_CORE)
