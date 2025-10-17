/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 16, 2023.
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

#if PLATFORM(MAC)
#import "ValidationBubble.h"

#import <AppKit/AppKit.h>
#import <wtf/text/WTFString.h>

@interface WebValidationPopover : NSPopover
@end

@implementation WebValidationPopover

- (void)mouseDown:(NSEvent *)event
{
    UNUSED_PARAM(event);
    [self close];
}

@end

namespace WebCore {

static const CGFloat horizontalPadding = 5;
static const CGFloat verticalPadding = 5;
static const CGFloat maxLabelWidth = 300;

ValidationBubble::ValidationBubble(NSView* view, const String& message, const Settings& settings)
    : m_view(view)
    , m_message(message)
{
    RetainPtr<NSViewController> controller = adoptNS([[NSViewController alloc] init]);

    RetainPtr<NSView> popoverView = adoptNS([[NSView alloc] initWithFrame:NSZeroRect]);
    [controller setView:popoverView.get()];

    RetainPtr<NSTextField> label = adoptNS([[NSTextField alloc] init]);
    [label setEditable:NO];
    [label setDrawsBackground:NO];
    [label setBordered:NO];
    [label setStringValue:message];
    m_fontSize = std::max(settings.minimumFontSize, 13.0);
    [label setFont:[NSFont systemFontOfSize:m_fontSize]];
    [label setMaximumNumberOfLines:4];
    [[label cell] setTruncatesLastVisibleLine:YES];
    [popoverView addSubview:label.get()];
    NSSize labelSize = [label sizeThatFits:NSMakeSize(maxLabelWidth, CGFLOAT_MAX)];
    [label setFrame:NSMakeRect(horizontalPadding, verticalPadding, labelSize.width, labelSize.height)];
    [popoverView setFrame:NSMakeRect(0, 0, labelSize.width + horizontalPadding * 2, labelSize.height + verticalPadding * 2)];

    m_popover = adoptNS([[WebValidationPopover alloc] init]);
    [m_popover setContentViewController:controller.get()];
    [m_popover setBehavior:NSPopoverBehaviorTransient];
    [m_popover setAnimates:NO];
}

ValidationBubble::~ValidationBubble()
{
    [m_popover close];
}

void ValidationBubble::showRelativeTo(const IntRect& anchorRect)
{
    // Passing an unparented view to [m_popover showRelativeToRect:ofView:preferredEdge:] crashes.
    if (![m_view window])
        return;

    NSRect rect = NSMakeRect(anchorRect.x(), anchorRect.y(), anchorRect.width(), anchorRect.height());
    [m_popover showRelativeToRect:rect ofView:m_view preferredEdge:NSMinYEdge];
}

} // namespace WebCore

#endif // PLATFORM(MAC)
