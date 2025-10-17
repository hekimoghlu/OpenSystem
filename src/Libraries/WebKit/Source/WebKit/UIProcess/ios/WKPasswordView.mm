/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 23, 2024.
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
#import "WKPasswordView.h"

#if PLATFORM(IOS_FAMILY)

#import "UIKitSPI.h"
#import "UIKitUtilities.h"
#import "WKContentView.h"
#import "WKWebViewContentProvider.h"
#import <WebCore/LocalizedStrings.h>
#import <wtf/RetainPtr.h>
#import <wtf/text/WTFString.h>

const CGFloat passwordEntryFieldPadding = 10;

@interface WKPasswordView () <UIDocumentPasswordViewDelegate>
@end

@implementation WKPasswordView {
    RetainPtr<NSString> _documentName;
    RetainPtr<UIScrollView> _scrollView;
    RetainPtr<UIDocumentPasswordView> _passwordView;
    CGFloat _savedMinimumZoomScale;
    CGFloat _savedMaximumZoomScale;
    CGFloat _savedZoomScale;
    CGSize _savedContentSize;
    RetainPtr<UIColor> _savedBackgroundColor;
}

- (instancetype)initWithFrame:(CGRect)frame documentName:(NSString *)documentName
{
    self = [super initWithFrame:frame];
    if (!self)
        return nil;

    _documentName = adoptNS([documentName copy]);
    _passwordView = adoptNS([[UIDocumentPasswordView alloc] initWithDocumentName:_documentName.get()]);
    [_passwordView setFrame:self.bounds];
    [_passwordView setPasswordDelegate:self];
    [_passwordView setAutoresizingMask:UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight];

    self.autoresizesSubviews = YES;
    [self addSubview:_passwordView.get()];

    return self;
}

- (void)dealloc
{
    [_userDidEnterPassword release];
    [super dealloc];
}

- (NSString *)documentName
{
    return _documentName.get();
}

- (void)layoutSubviews
{
    if (_scrollView)
        [_scrollView setContentSize:self.frame.size];
}

- (void)showInScrollView:(UIScrollView *)scrollView
{
    _scrollView = scrollView;

    _savedMinimumZoomScale = [_scrollView minimumZoomScale];
    _savedMaximumZoomScale = [_scrollView maximumZoomScale];
    _savedZoomScale = [_scrollView zoomScale];
    _savedContentSize = [_scrollView contentSize];
    _savedBackgroundColor = [_scrollView backgroundColor];

    [_scrollView setMinimumZoomScale:1];
    [_scrollView setMaximumZoomScale:1];
    [_scrollView setZoomScale:1];
    [_scrollView setContentSize:self.frame.size];
    [_scrollView setBackgroundColor:UIColor.systemGroupedBackgroundColor];

    [scrollView addSubview:self];
}

- (void)hide
{
    [_scrollView setMinimumZoomScale:_savedMinimumZoomScale];
    [_scrollView setMaximumZoomScale:_savedMaximumZoomScale];
    [_scrollView setZoomScale:_savedZoomScale];
    [_scrollView setContentSize:_savedContentSize];
    [_scrollView setBackgroundColor:_savedBackgroundColor.get()];

    _scrollView = nil;
    _savedBackgroundColor = nil;

    [self removeFromSuperview];
}

- (void)showPasswordFailureAlert
{
    [[_passwordView passwordField] setText:@""];
    auto alert = WebKit::createUIAlertController(WEB_UI_STRING("The document could not be opened with that password.", "document password failure alert message"), @"");

    UIAlertAction *defaultAction = [UIAlertAction actionWithTitle:WEB_UI_STRING_KEY("OK", "OK (password failure alert)", "OK button label in document password failure alert") style:UIAlertActionStyleDefault handler:[](UIAlertAction *) { }];

    [alert addAction:defaultAction];

    [self.window.rootViewController presentViewController:alert.get() animated:YES completion:nil];
}

- (void)_keyboardDidShow:(NSNotification *)notification
{
    UITextField *passwordField = [_passwordView passwordField];
    if (!passwordField.isEditing)
        return;

    CGRect keyboardRect = [UIPeripheralHost visiblePeripheralFrame];
    if (CGRectIsEmpty(keyboardRect))
        return;

    UIWindow *window = [_scrollView window];
    keyboardRect = [window convertRect:keyboardRect fromWindow:nil];
    keyboardRect = [_scrollView convertRect:keyboardRect fromView:window];

    CGRect passwordFieldFrame = [passwordField convertRect:passwordField.bounds toView:_scrollView.get()];

    CGSize contentSize = [_passwordView bounds].size;
    contentSize.height += CGRectGetHeight(keyboardRect);
    [_scrollView setContentSize:contentSize];

    if (CGRectIntersectsRect(passwordFieldFrame, keyboardRect)) {
        CGFloat yDelta = CGRectGetMaxY(passwordFieldFrame) - CGRectGetMinY(keyboardRect);

        CGPoint contentOffset = [_scrollView contentOffset];
        contentOffset.y += yDelta + passwordEntryFieldPadding;

        [_scrollView setContentOffset:contentOffset animated:YES];
    }
}

#pragma mark UIDocumentPasswordViewDelegate

- (void)userDidEnterPassword:(NSString *)password forPasswordView:(UIDocumentPasswordView *)passwordView
{
    auto protectedSelf = retainPtr(self);
    if (_userDidEnterPassword)
        _userDidEnterPassword(password);
}

- (void)didBeginEditingPassword:(UITextField *)passwordField inView:(UIDocumentPasswordView *)passwordView
{
    [[NSNotificationCenter defaultCenter] addObserver:self selector:@selector(_keyboardDidShow:) name:UIKeyboardDidShowNotification object:nil];
}

- (void)didEndEditingPassword:(UITextField *)passwordField inView:(UIDocumentPasswordView *)passwordView
{
    [_scrollView setContentSize:[_passwordView frame].size];

    UIEdgeInsets contentInset = [_scrollView contentInset];
    [_scrollView setContentOffset:CGPointMake(-contentInset.left, -contentInset.top) animated:YES];

    [[NSNotificationCenter defaultCenter] removeObserver:self name:UIKeyboardDidShowNotification object:nil];
}

@end

#endif // PLATFORM(IOS_FAMILY)
