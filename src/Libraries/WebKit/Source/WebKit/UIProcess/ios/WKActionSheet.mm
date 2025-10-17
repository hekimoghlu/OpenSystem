/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 10, 2021.
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
#import "WKActionSheet.h"

#if PLATFORM(IOS_FAMILY)

#import "UIKitSPI.h"
#import "UIKitUtilities.h"
#import <wtf/RetainPtr.h>

@implementation WKActionSheet {
    id <WKActionSheetDelegate> _sheetDelegate;
    UIPopoverArrowDirection _arrowDirections;
    BOOL _isRotating;
    BOOL _readyToPresentAfterRotation;

    WKActionSheetPresentationStyle _currentPresentationStyle;
    RetainPtr<UIViewController> _currentPresentingViewController;
    RetainPtr<UIViewController> _presentedViewControllerWhileRotating;
    RetainPtr<id <UIPopoverPresentationControllerDelegate>> _popoverPresentationControllerDelegateWhileRotating;
}

- (id)init
{
    self = [super init];
    if (!self)
        return nil;

    _arrowDirections = UIPopoverArrowDirectionAny;

ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    if (UI_USER_INTERFACE_IDIOM() != UIUserInterfaceIdiomPhone) {
        // Only iPads support popovers that rotate. UIActionSheets actually block rotation on iPhone/iPod Touch
        NSNotificationCenter *center = [NSNotificationCenter defaultCenter];
        [center addObserver:self selector:@selector(willRotate) name:UIWindowWillRotateNotification object:nil];
        [center addObserver:self selector:@selector(didRotate) name:UIWindowDidRotateNotification object:nil];
    }
ALLOW_DEPRECATED_DECLARATIONS_END

    return self;
}

- (void)dealloc
{
    [self _cleanup];
    [super dealloc];
}

- (void)_cleanup
{
    [[NSNotificationCenter defaultCenter] removeObserver:self];
    [NSObject cancelPreviousPerformRequestsWithTarget:self];
}

#pragma mark - Sheet presentation code

- (BOOL)presentSheet:(WKActionSheetPresentationStyle)style
{
    // Calculate the presentation rect just before showing.
    CGRect presentationRect = CGRectZero;
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    if (UI_USER_INTERFACE_IDIOM() != UIUserInterfaceIdiomPhone) {
        presentationRect = [self _presentationRectForStyle:style];
        if (CGRectIsEmpty(presentationRect))
            return NO;
    }
ALLOW_DEPRECATED_DECLARATIONS_END

    _currentPresentationStyle = style;
    return [self presentSheetFromRect:presentationRect];
}

- (CGRect)_presentationRectForStyle:(WKActionSheetPresentationStyle)style
{
    if (style == WKActionSheetPresentAtElementRect)
        return [_sheetDelegate presentationRectForIndicatedElement];

    if (style == WKActionSheetPresentAtClosestIndicatorRect)
        return [_sheetDelegate presentationRectForElementUsingClosestIndicatedRect];

    return [_sheetDelegate initialPresentationRectInHostViewForSheet];
}

- (BOOL)presentSheetFromRect:(CGRect)presentationRect
{
    UIView *view = [_sheetDelegate hostViewForSheet];
    if (!view)
        return NO;

    UIViewController *presentedViewController = _presentedViewControllerWhileRotating.get() ? _presentedViewControllerWhileRotating.get() : self;
    presentedViewController.modalPresentationStyle = UIModalPresentationPopover;

    UIPopoverPresentationController *presentationController = presentedViewController.popoverPresentationController;
    presentationController.sourceView = view;
    presentationController.sourceRect = presentationRect;
    presentationController.permittedArrowDirections = _arrowDirections;

    if (_popoverPresentationControllerDelegateWhileRotating)
        presentationController.delegate = _popoverPresentationControllerDelegateWhileRotating.get();

    _currentPresentingViewController = view._wk_viewControllerForFullScreenPresentation;
    [_currentPresentingViewController presentViewController:presentedViewController animated:YES completion:nil];

    return YES;
}

- (void)doneWithSheet:(BOOL)dismiss
{
    if (dismiss) {
        UIViewController *currentPresentedViewController = [_currentPresentingViewController presentedViewController];
        if (currentPresentedViewController == self || currentPresentedViewController == _presentedViewControllerWhileRotating)
            [currentPresentedViewController dismissViewControllerAnimated:YES completion:nil];
    }

    _currentPresentingViewController = nil;
    _presentedViewControllerWhileRotating = nil;
    _popoverPresentationControllerDelegateWhileRotating = nil;
    _currentPresentationStyle = WKActionSheetPresentAtTouchLocation;

    [self _cleanup];
}

#pragma mark - Rotation handling code

- (void)willRotate
{
    // We want to save the view controller that is currently being presented to re-present it after rotation.
    // Here are the various possible states that we have to handle:
    // a) topViewController presenting ourselves (alertViewController) -> nominal case.
    //    There is no need to save the presented view controller, which is self.
    // b) topViewController presenting ourselves presenting a content view controller ->
    //    This happens if one of the actions in the action sheet presented a different view controller inside the popover,
    //    using a current context presentation. This is for example the case with the Data Detectors action "Add to Contacts".
    // c) topViewController presenting that content view controller directly.
    //    This happens if we were in the (b) case and then rotated the device. Since we dismiss the popover during the
    //    rotation, we take this opportunity to simplify the view controller hierarchy and simply re-present the content
    //    view controller, without re-presenting the alert controller.

    UIView *view = [_sheetDelegate hostViewForSheet];
    if (!view)
        return;

    auto presentingViewController = view._wk_viewControllerForFullScreenPresentation;

    // topPresentedViewController is either self (cases (a) and (b) above) or an action's view controller
    // (case (c) above).
    UIViewController *topPresentedViewController = [presentingViewController presentedViewController];

    // We only have something to do if we're showing a popover (that we have to reposition).
    // Otherwise the default UIAlertController behaviour is enough.
    if ([topPresentedViewController presentationController].presentationStyle != UIModalPresentationPopover)
        return;

    if (_isRotating)
        return;

    _isRotating = YES;
    _readyToPresentAfterRotation = NO;

    UIViewController *presentedViewController = nil;
    if ([self presentingViewController] != nil) {
        // Handle cases (a) and (b) above (we (UIAlertController) are still in the presentation hierarchy).
        // Save the view controller presented by one of the actions if there is one.
        // (In the (a) case, presentedViewController will be nil).

        presentedViewController = [self presentedViewController];
    } else {
        // Handle case (c) above.
        // The view controller that we want to save is the top presented view controller, since we
        // are not presenting it anymore.

        presentedViewController = topPresentedViewController;
    }

    _presentedViewControllerWhileRotating = presentedViewController;

    // Save the popover presentation controller's delegate, because in case (b) we're going to use
    // a different popoverPresentationController after rotation to re-present the action view controller,
    // and that action is still expecting delegate callbacks when the popover is dismissed.
    _popoverPresentationControllerDelegateWhileRotating = [topPresentedViewController popoverPresentationController].delegate;

    [presentingViewController dismissViewControllerAnimated:NO completion:^{
        [self updateSheetPosition];
    }];
}

- (void)updateSheetPosition
{
    UIViewController *presentedViewController = _presentedViewControllerWhileRotating.get() ? _presentedViewControllerWhileRotating.get() : self;

    // There are two asynchronous events which might trigger this call, and we have to wait for both of them before doing something.
    // - One runloop iteration after rotation (to let the Web content re-layout, see below)
    // - The completion of the view controller dismissal in willRotate.
    // (We cannot present something again until the dismissal is done)

    if (_isRotating || !_readyToPresentAfterRotation || [presentedViewController presentingViewController] || [self presentingViewController])
        return;

    CGRect presentationRect = [self _presentationRectForStyle:_currentPresentationStyle];
    BOOL wasPresentedViewControllerModal = [_presentedViewControllerWhileRotating isModalInPresentation];

    if (!CGRectIsEmpty(presentationRect) || wasPresentedViewControllerModal) {
        // Re-present the popover only if we are still pointing to content onscreen, or if we can't dismiss it without losing information.
        // (if the view controller is modal)

        CGRect intersection = CGRectIntersection([[_sheetDelegate hostViewForSheet] bounds], presentationRect);
        if (!CGRectIsEmpty(intersection))
            [self presentSheetFromRect:intersection];
        else if (wasPresentedViewControllerModal)
            [self presentSheet:_currentPresentationStyle];

        _presentedViewControllerWhileRotating = nil;
        _popoverPresentationControllerDelegateWhileRotating = nil;
    }
}

- (void)_didRotateAndLayout
{
    _isRotating = NO;
    _readyToPresentAfterRotation = YES;
    [_sheetDelegate updatePositionInformation];
    [self updateSheetPosition];
}

- (void)didRotate
{
    // Handle the rotation on the next run loop interation as this
    // allows the onOrientationChange event to fire, and the element node may
    // be removed.
    // <rdar://problem/9360929> Should re-present popover after layout rather than on the next runloop

    [self performSelector:@selector(_didRotateAndLayout) withObject:nil afterDelay:0];
}

@end

#endif // PLATFORM(IOS_FAMILY)
