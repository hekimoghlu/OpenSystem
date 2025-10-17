/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 30, 2025.
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

#if !PLATFORM(IOS_FAMILY)

#import "WebCoreFullScreenWindow.h"

// FIXME: This isn't really an NSWindowController method - it's a method that
// the NSWindowController subclass that's using WebCoreFullScreenWindow needs to implement.
// It should probably be a protocol method.
@interface NSWindowController ()
- (void)performClose:(id)sender;
@end

@implementation WebCoreFullScreenWindow

- (id)initWithContentRect:(NSRect)contentRect styleMask:(NSUInteger)aStyle backing:(NSBackingStoreType)bufferingType defer:(BOOL)flag
{
    self = [super initWithContentRect:contentRect styleMask:aStyle backing:bufferingType defer:flag];
    if (!self)
        return nil;
    [self setOpaque:NO];
    [self setBackgroundColor:[NSColor clearColor]];
    [self setIgnoresMouseEvents:NO];
    [self setAcceptsMouseMovedEvents:YES];
    [self setReleasedWhenClosed:NO];
    [self setHasShadow:NO];

    return self;
}

- (NSRect)constrainFrameRect:(NSRect)frameRect toScreen:(NSScreen *)screen
{
    UNUSED_PARAM(screen);
    return frameRect;
}

- (BOOL)canBecomeMainWindow
{
    return NO;
}

- (BOOL)canBecomeKeyWindow
{
    return YES;
}

- (void)keyDown:(NSEvent *)theEvent
{
    if ([[theEvent charactersIgnoringModifiers] isEqual:@"\e"]) // Esacpe key-code
        [self cancelOperation:self];
    else [super keyDown:theEvent];
}

- (void)cancelOperation:(id)sender
{
    [[self windowController] cancelOperation:sender];
}

- (void)performClose:(id)sender
{
    [[self windowController] performClose:sender];
}

- (void)setStyleMask:(NSUInteger)styleMask
{
    // Changing the styleMask of a NSWindow can reset the firstResponder if the frame view changes,
    // so save off the existing one, and restore it if necessary after the call to -setStyleMask:.
    NSResponder* savedFirstResponder = [self firstResponder];

    [super setStyleMask:styleMask];

    if ([self firstResponder] != savedFirstResponder
        && [savedFirstResponder isKindOfClass:[NSView class]]
        && [(NSView*)savedFirstResponder isDescendantOf:[self contentView]])
        [self makeFirstResponder:savedFirstResponder];
}
@end

#endif // !PLATFORM(IOS_FAMILY)
