/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 17, 2024.
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
#import "WKBaseScrollView.h"

#if PLATFORM(IOS_FAMILY)

#import "RemoteLayerTreeViews.h"
#import "UIKitSPI.h"
#import "WKContentView.h"
#import <objc/runtime.h>
#import <wtf/RetainPtr.h>
#import <wtf/RuntimeApplicationChecks.h>
#import <wtf/SetForScope.h>
#import <wtf/WeakObjCPtr.h>
#import <wtf/cocoa/RuntimeApplicationChecksCocoa.h>

#if ENABLE(OVERLAY_REGIONS_IN_EVENT_REGION)
#import "UIKitUtilities.h"
#import <pal/spi/cocoa/QuartzCoreSPI.h>
#import <wtf/cocoa/VectorCocoa.h>
#endif

@interface UIScrollView (GestureRecognizerDelegate) <UIGestureRecognizerDelegate>
@end

@implementation WKBaseScrollView {
    RetainPtr<UIPanGestureRecognizer> _axisLockingPanGestureRecognizer;
    UIAxis _axesToPreventMomentumScrolling;
    BOOL _isBeingRemovedFromSuperview;
#if ENABLE(OVERLAY_REGIONS_IN_EVENT_REGION)
    HashSet<WebCore::IntRect> _overlayRegionRects;
#endif
}

- (instancetype)initWithFrame:(CGRect)frame
{
    [WKBaseScrollView _overrideAddGestureRecognizerIfNeeded];

    if (!(self = [super initWithFrame:frame]))
        return nil;

#if HAVE(UISCROLLVIEW_ASYNCHRONOUS_SCROLL_EVENT_HANDLING) && !USE(BROWSERENGINEKIT)
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    self._allowsAsyncScrollEvent = YES;
ALLOW_DEPRECATED_DECLARATIONS_END
#endif

    _axesToPreventMomentumScrolling = UIAxisNeither;
    [self.panGestureRecognizer addTarget:self action:@selector(_updatePanGestureToPreventScrolling)];
    return self;
}

+ (void)_overrideAddGestureRecognizerIfNeeded
{
    static bool hasOverridenAddGestureRecognizer = false;
    if (std::exchange(hasOverridenAddGestureRecognizer, true))
        return;

    if (WTF::IOSApplication::isHimalaya() && !linkedOnOrAfterSDKWithBehavior(SDKAlignedBehavior::ScrollViewSubclassImplementsAddGestureRecognizer)) {
        // This check can be removed and -_wk_addGestureRecognizer: can be renamed to -addGestureRecognizer: once the å–œé©¬æ‹‰é›… app updates to a version of
        // the iOS 17 SDK with this WKBaseScrollView refactoring. Otherwise, the call to `-[super addGestureRecognizer:]` below will fail, due to how this
        // app uses `class_getInstanceMethod` and `method_setImplementation` to intercept and override all calls to `-[UIView addGestureRecognizer:]`.
        return;
    }

    auto method = class_getInstanceMethod(self.class, @selector(_wk_addGestureRecognizer:));
    class_addMethod(self.class, @selector(addGestureRecognizer:), method_getImplementation(method), method_getTypeEncoding(method));
}

- (void)_wk_addGestureRecognizer:(UIGestureRecognizer *)gestureRecognizer
{
    if (self.panGestureRecognizer == gestureRecognizer) {
        if (!_axisLockingPanGestureRecognizer) {
            _axisLockingPanGestureRecognizer = adoptNS([[UIPanGestureRecognizer alloc] initWithTarget:self action:@selector(_updatePanGestureToPreventScrolling)]);
            [_axisLockingPanGestureRecognizer setName:@"Scroll axis locking"];
            [_axisLockingPanGestureRecognizer setDelegate:self];
        }
        [self addGestureRecognizer:_axisLockingPanGestureRecognizer.get()];
    }

    [super addGestureRecognizer:gestureRecognizer];
}

- (void)removeGestureRecognizer:(UIGestureRecognizer *)gestureRecognizer
{
    if (self.panGestureRecognizer == gestureRecognizer) {
        if (auto gesture = std::exchange(_axisLockingPanGestureRecognizer, nil))
            [self removeGestureRecognizer:gesture.get()];
    }

    [super removeGestureRecognizer:gestureRecognizer];
}

- (void)_updatePanGestureToPreventScrolling
{
    auto panGesture = self.panGestureRecognizer;
    switch (self.panGestureRecognizer.state) {
    case UIGestureRecognizerStatePossible:
    case UIGestureRecognizerStateEnded:
    case UIGestureRecognizerStateCancelled:
    case UIGestureRecognizerStateFailed:
        return;
    case UIGestureRecognizerStateBegan:
    case UIGestureRecognizerStateChanged:
        break;
    }

    switch ([_axisLockingPanGestureRecognizer state]) {
    case UIGestureRecognizerStateCancelled:
    case UIGestureRecognizerStateFailed:
        return;
    case UIGestureRecognizerStatePossible:
    case UIGestureRecognizerStateBegan:
    case UIGestureRecognizerStateChanged:
    case UIGestureRecognizerStateEnded:
        break;
    }

    auto axesToPrevent = self._axesToPreventScrollingFromDelegate;
    if (axesToPrevent == UIAxisNeither)
        return;

    auto adjustedTranslation = [panGesture translationInView:nil];
    bool translationChanged = false;
    if ((axesToPrevent & UIAxisHorizontal) && std::abs(adjustedTranslation.x) > CGFLOAT_EPSILON) {
        adjustedTranslation.x = 0;
        _axesToPreventMomentumScrolling |= UIAxisHorizontal;
        translationChanged = true;
    }

    if ((axesToPrevent & UIAxisVertical) && std::abs(adjustedTranslation.y) > CGFLOAT_EPSILON) {
        adjustedTranslation.y = 0;
        _axesToPreventMomentumScrolling |= UIAxisVertical;
        translationChanged = true;
    }

    if (translationChanged)
        [panGesture setTranslation:adjustedTranslation inView:nil];
}

- (void)removeFromSuperview
{
    SetForScope removeFromSuperviewScope { _isBeingRemovedFromSuperview, YES };

    [super removeFromSuperview];
}

- (UIAxis)_axesToPreventScrollingFromDelegate
{
    if (_isBeingRemovedFromSuperview || !self.window)
        return UIAxisNeither;
    auto delegate = self.baseScrollViewDelegate;
    return delegate ? [delegate axesToPreventScrollingForPanGestureInScrollView:self] : UIAxisNeither;
}


#if ENABLE(OVERLAY_REGIONS_IN_EVENT_REGION)

#if USE(APPLE_INTERNAL_SDK) && __has_include(<WebKitAdditions/WKBaseScrollViewAdditions.mm>)
#import <WebKitAdditions/WKBaseScrollViewAdditions.mm>
#else
- (BOOL)_hasEnoughContentForOverlayRegions { return false; }
- (void)_updateOverlayRegionsBehavior:(BOOL)selected { }
- (void)_updateOverlayRegionRects:(const HashSet<WebCore::IntRect>&)overlayRegions { }
- (void)_updateOverlayRegions:(NSArray<NSData *> *)overlayRegions { }
#endif

#endif // ENABLE(OVERLAY_REGIONS_IN_EVENT_REGION)

#pragma mark - UIGestureRecognizerDelegate

- (BOOL)gestureRecognizer:(UIGestureRecognizer *)gestureRecognizer shouldRecognizeSimultaneouslyWithGestureRecognizer:(UIGestureRecognizer *)otherGestureRecognizer
{
    if (gestureRecognizer == _axisLockingPanGestureRecognizer || otherGestureRecognizer == _axisLockingPanGestureRecognizer)
        return YES;

    static BOOL callIntoSuperclass = [UIScrollView instancesRespondToSelector:@selector(gestureRecognizer:shouldRecognizeSimultaneouslyWithGestureRecognizer:)];
    if (!callIntoSuperclass)
        return NO;

    return [super gestureRecognizer:gestureRecognizer shouldRecognizeSimultaneouslyWithGestureRecognizer:otherGestureRecognizer];
}

- (BOOL)gestureRecognizerShouldBegin:(UIGestureRecognizer *)gestureRecognizer
{
    if (self.panGestureRecognizer == gestureRecognizer)
        _axesToPreventMomentumScrolling = UIAxisNeither;

    static BOOL callIntoSuperclass = [UIScrollView instancesRespondToSelector:@selector(gestureRecognizerShouldBegin:)];
    if (!callIntoSuperclass)
        return YES;

    return [super gestureRecognizerShouldBegin:gestureRecognizer];
}

- (BOOL)gestureRecognizer:(UIGestureRecognizer *)gestureRecognizer shouldReceiveTouch:(UITouch *)touch
{
    if (self.panGestureRecognizer == gestureRecognizer) {
        RetainPtr delegate = [self baseScrollViewDelegate];
        if (delegate && ![delegate shouldAllowPanGestureRecognizerToReceiveTouchesInScrollView:self])
            return NO;
    }

    static BOOL callIntoSuperclass = [UIScrollView instancesRespondToSelector:@selector(gestureRecognizer:shouldReceiveTouch:)];
    if (!callIntoSuperclass)
        return YES;

    return [super gestureRecognizer:gestureRecognizer shouldReceiveTouch:touch];
}

@end

#endif // PLATFORM(IOS_FAMILY)
