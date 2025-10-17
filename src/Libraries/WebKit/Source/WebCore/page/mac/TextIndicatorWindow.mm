/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 25, 2022.
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
#import "TextIndicatorWindow.h"

#if PLATFORM(MAC)

#import "GeometryUtilities.h"
#import "GraphicsContext.h"
#import "PathUtilities.h"
#import "TextIndicator.h"
#import "WebActionDisablingCALayerDelegate.h"
#import "WebTextIndicatorLayer.h"
#import <pal/spi/cg/CoreGraphicsSPI.h>
#import <pal/spi/cocoa/QuartzCoreSPI.h>
#import <pal/spi/mac/NSColorSPI.h>
#import <wtf/TZoneMallocInlines.h>

@interface WebTextIndicatorView : NSView

@end

@implementation WebTextIndicatorView

- (BOOL)isFlipped
{
    return YES;
}

@end


using namespace WebCore;

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(TextIndicatorWindow);

TextIndicatorWindow::TextIndicatorWindow(NSView *targetView)
    : m_targetView(targetView)
    , m_temporaryTextIndicatorTimer(RunLoop::main(), this, &TextIndicatorWindow::startFadeOut)
{
}

TextIndicatorWindow::~TextIndicatorWindow()
{
    clearTextIndicator(WebCore::TextIndicatorDismissalAnimation::FadeOut);
}

void TextIndicatorWindow::setAnimationProgress(float progress)
{
    if (!m_textIndicator)
        return;

    [m_textIndicatorLayer setAnimationProgress:progress];
}

void TextIndicatorWindow::clearTextIndicator(WebCore::TextIndicatorDismissalAnimation animation)
{
    RefPtr<TextIndicator> textIndicator = WTFMove(m_textIndicator);

    if ([m_textIndicatorLayer isFadingOut])
        return;

    if (textIndicator && [m_textIndicatorLayer indicatorWantsManualAnimation:*textIndicator] && [m_textIndicatorLayer hasCompletedAnimation] && animation == WebCore::TextIndicatorDismissalAnimation::FadeOut) {
        startFadeOut();
        return;
    }

    closeWindow();
}

void TextIndicatorWindow::setTextIndicator(Ref<TextIndicator> textIndicator, CGRect textBoundingRectInScreenCoordinates, TextIndicatorLifetime lifetime)
{
    if (m_textIndicator == textIndicator.ptr())
        return;

    closeWindow();

    m_textIndicator = textIndicator.ptr();

    CGFloat horizontalMargin = dropShadowBlurRadius * 2 + TextIndicator::defaultHorizontalMargin;
    CGFloat verticalMargin = dropShadowBlurRadius * 2 + TextIndicator::defaultVerticalMargin;
    
    if ([m_textIndicatorLayer indicatorWantsBounce:*m_textIndicator]) {
        horizontalMargin = std::max(horizontalMargin, textBoundingRectInScreenCoordinates.size.width * (WebCore::midBounceScale - 1) + horizontalMargin);
        verticalMargin = std::max(verticalMargin, textBoundingRectInScreenCoordinates.size.height * (WebCore::midBounceScale - 1) + verticalMargin);
    }

    horizontalMargin = CGCeiling(horizontalMargin);
    verticalMargin = CGCeiling(verticalMargin);

    CGRect contentRect = CGRectInset(textBoundingRectInScreenCoordinates, -horizontalMargin, -verticalMargin);
    NSRect windowContentRect = [NSWindow contentRectForFrameRect:NSRectFromCGRect(contentRect) styleMask:NSWindowStyleMaskBorderless];
    NSRect integralWindowContentRect = NSIntegralRect(windowContentRect);
    NSPoint fractionalTextOffset = NSMakePoint(windowContentRect.origin.x - integralWindowContentRect.origin.x, windowContentRect.origin.y - integralWindowContentRect.origin.y);
    m_textIndicatorWindow = adoptNS([[NSWindow alloc] initWithContentRect:integralWindowContentRect styleMask:NSWindowStyleMaskBorderless backing:NSBackingStoreBuffered defer:NO]);
    [m_textIndicatorWindow setBackgroundColor:[NSColor clearColor]];
    [m_textIndicatorWindow setOpaque:NO];
    [m_textIndicatorWindow setIgnoresMouseEvents:YES];

    NSRect frame = NSMakeRect(0, 0, [m_textIndicatorWindow frame].size.width, [m_textIndicatorWindow frame].size.height);
    m_textIndicatorLayer = adoptNS([[WebTextIndicatorLayer alloc] initWithFrame:frame
        textIndicator:*m_textIndicator margin:NSMakeSize(horizontalMargin, verticalMargin) offset:fractionalTextOffset]);
    m_textIndicatorView = adoptNS([[WebTextIndicatorView alloc] initWithFrame:frame]);
    [m_textIndicatorView setLayer:m_textIndicatorLayer.get()];
    [m_textIndicatorView setWantsLayer:YES];
    [m_textIndicatorWindow setContentView:m_textIndicatorView.get()];

    [[m_targetView window] addChildWindow:m_textIndicatorWindow.get() ordered:NSWindowAbove];
    [m_textIndicatorWindow setReleasedWhenClosed:NO];

    if (m_textIndicator->presentationTransition() != TextIndicatorPresentationTransition::None)
        [m_textIndicatorLayer present];

    if (lifetime == TextIndicatorLifetime::Temporary)
        m_temporaryTextIndicatorTimer.startOneShot(WebCore::timeBeforeFadeStarts);
}

void TextIndicatorWindow::closeWindow()
{
    if (!m_textIndicatorWindow)
        return;

    if ([m_textIndicatorLayer isFadingOut])
        return;

    m_temporaryTextIndicatorTimer.stop();

    [[m_textIndicatorWindow parentWindow] removeChildWindow:m_textIndicatorWindow.get()];
    [m_textIndicatorWindow close];
    m_textIndicatorWindow = nullptr;
}

void TextIndicatorWindow::startFadeOut()
{
    [m_textIndicatorLayer setFadingOut:YES];
    RetainPtr<NSWindow> indicatorWindow = m_textIndicatorWindow;
    [m_textIndicatorLayer hideWithCompletionHandler:[indicatorWindow] {
        [[indicatorWindow parentWindow] removeChildWindow:indicatorWindow.get()];
        [indicatorWindow close];
    }];
}

} // namespace WebCore

#endif // PLATFORM(MAC)
