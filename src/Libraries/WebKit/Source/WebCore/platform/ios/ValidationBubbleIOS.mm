/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 31, 2023.
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

#if PLATFORM(IOS_FAMILY)

#import "ValidationBubble.h"

#import "UIViewControllerUtilities.h"
#import <UIKit/UIGeometry.h>
#import <objc/message.h>
#import <pal/ios/UIKitSoftLink.h>
#import <pal/spi/ios/UIKitSPI.h>
#import <wtf/RetainPtr.h>
#import <wtf/SoftLinking.h>
#import <wtf/text/WTFString.h>

// Add a bit of vertical and horizontal padding between the
// label and its parent view, to avoid laying out the label
// against the edges of the popover view.
constexpr CGFloat validationBubbleHorizontalPadding = 17;
constexpr CGFloat validationBubbleVerticalPadding = 9;

// Avoid making the validation bubble too wide by enforcing a
// maximum width on the content size of the validation bubble
// view controller.
constexpr CGFloat validationBubbleMaxLabelWidth = 300;

// Avoid making the validation bubble too tall by truncating
// the label to a maximum of 4 lines.
constexpr NSInteger validationBubbleMaxNumberOfLines = 4;

@interface WebValidationBubbleViewController : UIViewController
@end

static const void* const validationBubbleViewControllerLabelKey = &validationBubbleViewControllerLabelKey;

static UILabel *label(WebValidationBubbleViewController *controller)
{
    return objc_getAssociatedObject(controller, validationBubbleViewControllerLabelKey);
}

static void updateLabelFrame(WebValidationBubbleViewController *controller)
{
    auto frameWithPadding = UIEdgeInsetsInsetRect(controller.view.bounds, controller.view.safeAreaInsets);
    label(controller).frame = UIEdgeInsetsInsetRect(frameWithPadding, UIEdgeInsetsMake(validationBubbleVerticalPadding, validationBubbleHorizontalPadding, validationBubbleVerticalPadding, validationBubbleHorizontalPadding));
}

static void callSuper(WebValidationBubbleViewController *instance, SEL selector)
{
    objc_super superStructure { instance, PAL::getUIViewControllerClass() };
    auto msgSendSuper = reinterpret_cast<void(*)(objc_super*, SEL)>(objc_msgSendSuper);
    msgSendSuper(&superStructure, selector);
}

static void WebValidationBubbleViewController_viewDidLoad(WebValidationBubbleViewController *instance, SEL)
{
    callSuper(instance, @selector(viewDidLoad));

    auto label = adoptNS([PAL::allocUILabelInstance() init]);
    [label setFont:[PAL::getUIFontClass() preferredFontForTextStyle:PAL::get_UIKit_UIFontTextStyleCallout()]];
    [label setLineBreakMode:NSLineBreakByTruncatingTail];
    [label setNumberOfLines:validationBubbleMaxNumberOfLines];
    [instance.view addSubview:label.get()];
    objc_setAssociatedObject(instance, validationBubbleViewControllerLabelKey, label.get(), OBJC_ASSOCIATION_RETAIN_NONATOMIC);
}

static void WebValidationBubbleViewController_viewWillLayoutSubviews(WebValidationBubbleViewController *instance, SEL)
{
    callSuper(instance, @selector(viewWillLayoutSubviews));
    updateLabelFrame(instance);
}

static void WebValidationBubbleViewController_viewSafeAreaInsetsDidChange(WebValidationBubbleViewController *instance, SEL)
{
    callSuper(instance, @selector(viewSafeAreaInsetsDidChange));
    updateLabelFrame(instance);
}

static WebValidationBubbleViewController *allocWebValidationBubbleViewControllerInstance()
{
    static Class theClass = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        theClass = objc_allocateClassPair(PAL::getUIViewControllerClass(), "WebValidationBubbleViewController", 0);
        class_addMethod(theClass, @selector(viewDidLoad), (IMP)WebValidationBubbleViewController_viewDidLoad, "v@:");
        class_addMethod(theClass, @selector(viewWillLayoutSubviews), (IMP)WebValidationBubbleViewController_viewWillLayoutSubviews, "v@:");
        class_addMethod(theClass, @selector(viewSafeAreaInsetsDidChange), (IMP)WebValidationBubbleViewController_viewSafeAreaInsetsDidChange, "v@:");
        objc_registerClassPair(theClass);
    });
    return (WebValidationBubbleViewController *)[theClass alloc];
}

@interface WebValidationBubbleTapRecognizer : NSObject
@end

@implementation WebValidationBubbleTapRecognizer {
    RetainPtr<UIViewController> _popoverController;
    RetainPtr<UITapGestureRecognizer> _tapGestureRecognizer;
}

- (WebValidationBubbleTapRecognizer *)initWithPopoverController:(UIViewController *)popoverController
{
    self = [super init];
    if (!self)
        return nil;

    _popoverController = popoverController;
    _tapGestureRecognizer = adoptNS([PAL::allocUITapGestureRecognizerInstance() initWithTarget:self action:@selector(dismissPopover)]);
    [[_popoverController view] addGestureRecognizer:_tapGestureRecognizer.get()];

    return self;
}

- (void)dealloc
{
    [[_popoverController view] removeGestureRecognizer:_tapGestureRecognizer.get()];
    [super dealloc];
}

- (void)dismissPopover
{
    [_popoverController dismissViewControllerAnimated:NO completion:nil];
}

@end

@interface WebValidationBubbleDelegate : NSObject <UIPopoverPresentationControllerDelegate> {
}
@end

@implementation WebValidationBubbleDelegate

- (UIModalPresentationStyle)adaptivePresentationStyleForPresentationController:(UIPresentationController *)controller traitCollection:(UITraitCollection *)traitCollection
{
    UNUSED_PARAM(controller);
    UNUSED_PARAM(traitCollection);
    // This is needed to force UIKit to use a popover on iPhone as well.
    return UIModalPresentationNone;
}

@end

namespace WebCore {

ValidationBubble::ValidationBubble(UIView *view, const String& message, const Settings&)
    : m_view(view)
    , m_message(message)
{
    m_popoverController = adoptNS([allocWebValidationBubbleViewControllerInstance() init]);
    [m_popoverController setModalPresentationStyle:UIModalPresentationPopover];
    m_tapRecognizer = adoptNS([[WebValidationBubbleTapRecognizer alloc] initWithPopoverController:m_popoverController.get()]);

    UILabel *validationLabel = label(m_popoverController.get());
    validationLabel.text = message;
    m_fontSize = validationLabel.font.pointSize;
    CGSize labelSize = [validationLabel sizeThatFits:CGSizeMake(validationBubbleMaxLabelWidth, CGFLOAT_MAX)];
    [m_popoverController setPreferredContentSize:CGSizeMake(labelSize.width + validationBubbleHorizontalPadding * 2, labelSize.height + validationBubbleVerticalPadding * 2)];
}

ValidationBubble::~ValidationBubble()
{
    [m_popoverController dismissViewControllerAnimated:NO completion:nil];
}

void ValidationBubble::show()
{
    if ([m_popoverController parentViewController] || [m_popoverController presentingViewController] || m_startingToPresentViewController)
        return;

    // Protect the validation bubble so it stays alive until it is effectively presented. UIKit does not deal nicely with
    // dismissing a popover that is being presented.
    RefPtr<ValidationBubble> protectedThis(this);
    m_startingToPresentViewController = true;
    [m_presentingViewController presentViewController:m_popoverController.get() animated:NO completion:[protectedThis]() {
        // Hide this popover from VoiceOver and instead announce the message.
        [protectedThis->m_popoverController view].accessibilityElementsHidden = YES;
        protectedThis->m_startingToPresentViewController = false;
    }];

    PAL::softLinkUIKitUIAccessibilityPostNotification(PAL::get_UIKit_UIAccessibilityAnnouncementNotification(), m_message);
}

static UIViewController *fallbackViewController(UIView *view)
{
    // FIXME: This logic to find a fallback view controller should move out of WebCore,
    // and into the client layer.
    for (UIView *currentView = view; currentView; currentView = currentView.superview) {
        if (auto controller = viewController(currentView))
            return controller;
    }
    NSLog(@"Failed to find a view controller to show form validation popover");
    return nil;
}

void ValidationBubble::setAnchorRect(const IntRect& anchorRect, UIViewController *presentingViewController)
{
    if (!presentingViewController)
        presentingViewController = fallbackViewController(m_view);

    if (!presentingViewController)
        return;

    UIPopoverPresentationController *presentationController = [m_popoverController popoverPresentationController];
    m_popoverDelegate = adoptNS([[WebValidationBubbleDelegate alloc] init]);
    presentationController.delegate = m_popoverDelegate.get();
    presentationController.passthroughViews = @[ presentingViewController.view, m_view ];
    presentationController.sourceView = m_view;
    presentationController.sourceRect = CGRectMake(anchorRect.x(), anchorRect.y(), anchorRect.width(), anchorRect.height());
    m_presentingViewController = presentingViewController;
}

} // namespace WebCore

#endif // PLATFORM(IOS_FAMILY)
