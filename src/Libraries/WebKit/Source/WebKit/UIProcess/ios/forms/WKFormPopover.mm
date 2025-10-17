/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 13, 2024.
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
#import "WKFormPopover.h"

#if PLATFORM(IOS_FAMILY)

#import "UIKitSPI.h"
#import "WKContentView.h"
#import "WKContentViewInteraction.h"
#import "WebPageProxy.h"
#import <wtf/RetainPtr.h>

using namespace WebKit;

ALLOW_DEPRECATED_DECLARATIONS_BEGIN

@implementation WKFormRotatingAccessoryPopover

- (id)initWithView:(WKContentView *)view
{
    if (!(self = [super initWithView:view]))
        return nil;
    self.dismissionDelegate = self;
    return self;
}

- (void)accessoryDone
{
    [self.view accessoryDone];
}

- (UIPopoverArrowDirection)popoverArrowDirections
{
    UIPopoverArrowDirection directions = UIPopoverArrowDirectionUp | UIPopoverArrowDirectionDown;
    if (UIInterfaceOrientationIsLandscape([[UIApplication sharedApplication] statusBarOrientation]) && [[UIPeripheralHost sharedInstance] isOnScreen])
        directions = UIPopoverArrowDirectionLeft | UIPopoverArrowDirectionRight;
    return directions;
}

- (void)popoverWasDismissed:(WKRotatingPopover *)popover
{
    [self accessoryDone];
}

@end

@interface WKRotatingPopover () <UIPopoverControllerDelegate>
@end

@implementation WKRotatingPopover {
    WKContentView *_view;

    BOOL _isRotating;
    BOOL _isPreservingFocus;
    CGPoint _presentationPoint;
    RetainPtr<UIPopoverController> _popoverController;
    id <WKRotatingPopoverDelegate> _dismissionDelegate;
}

- (id)initWithView:(WKContentView *)view
{
    if (!(self = [super init]))
        return nil;

    _view = view;
    self.presentationPoint = CGPointZero;

    NSNotificationCenter *center = [NSNotificationCenter defaultCenter];
    [center addObserver:self selector:@selector(willRotate:) name:UIWindowWillRotateNotification object:nil];
    [center addObserver:self selector:@selector(didRotate:) name:UIWindowDidRotateNotification object:nil];

    return self;
}

- (void)dealloc
{
    _view = nil;

    [_popoverController dismissPopoverAnimated:YES];
    [_popoverController setDelegate:nil];
    self.popoverController = nil;

    NSNotificationCenter *center = [NSNotificationCenter defaultCenter];
    [center removeObserver:self name:UIWindowWillRotateNotification object:nil];
    [center removeObserver:self name:UIWindowDidRotateNotification object:nil];

    [super dealloc];
}

- (UIPopoverController *)popoverController
{
    return _popoverController.get();
}

- (void)setPopoverController:(UIPopoverController *)popoverController
{
    if (_popoverController == popoverController)
        return;

    [_popoverController setDelegate:nil];
    _popoverController = popoverController;
    [_popoverController setDelegate:self];
}

- (UIPopoverArrowDirection)popoverArrowDirections
{
    return UIPopoverArrowDirectionAny;
}

- (void)presentPopoverAnimated:(BOOL)animated
{
    auto directions = [self popoverArrowDirections];
    CGRect presentationRect;
    if (CGPointEqualToPoint(self.presentationPoint, CGPointZero))
        presentationRect = _view.focusedElementInformation.interactionRect;
    else {
        auto scale = _view.page->pageScaleFactor();
        presentationRect = CGRectMake(self.presentationPoint.x * scale, self.presentationPoint.y * scale, 1, 1);
    }

    if (!CGRectIntersectsRect(presentationRect, _view.bounds))
        return;

#if PLATFORM(MACCATALYST)
    [_view startRelinquishingFirstResponderToFocusedElement];
#endif
    [_popoverController presentPopoverFromRect:CGRectIntegral(presentationRect) inView:_view permittedArrowDirections:directions animated:animated];
}

- (void)dismissPopoverAnimated:(BOOL)animated
{
#if PLATFORM(MACCATALYST)
    [_view stopRelinquishingFirstResponderToFocusedElement];
#endif

    [_popoverController dismissPopoverAnimated:animated];
}

- (void)willRotate:(NSNotification *)notification
{
    _isRotating = YES;
    [_popoverController dismissPopoverAnimated:NO];
}

- (void)didRotate:(NSNotification *)notification
{
    _isRotating = NO;
    [self presentPopoverAnimated:NO];
}

ALLOW_DEPRECATED_IMPLEMENTATIONS_BEGIN
- (void)popoverControllerDidDismissPopover:(UIPopoverController *)popoverController
ALLOW_DEPRECATED_IMPLEMENTATIONS_END
{
    if (_isRotating)
        return;

    [_dismissionDelegate popoverWasDismissed:self];
}

@end

ALLOW_DEPRECATED_DECLARATIONS_END

#endif // PLATFORM(IOS_FAMILY)
