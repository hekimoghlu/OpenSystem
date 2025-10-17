/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 27, 2022.
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
#import "WAKResponder.h"

#if PLATFORM(IOS_FAMILY)

#import "WAKViewInternal.h"
#import "WKViewPrivate.h"

@implementation WAKResponder

// FIXME: the functions named handleEvent generally do not forward event to the parent chain.
// This method should ideally be removed, or renamed "sendEvent".
- (void)handleEvent:(WebEvent *)event
{
    UNUSED_PARAM(event);
}

- (void)_forwardEvent:(WebEvent *)event
{
    [[self nextResponder] handleEvent:event];
}

- (void)scrollWheel:(WebEvent *)event 
{ 
    [self _forwardEvent:event];
}

- (void)mouseEntered:(WebEvent *)event
{ 
    [self _forwardEvent:event];
}

- (void)mouseExited:(WebEvent *)event 
{ 
    [self _forwardEvent:event];
}

- (void)mouseMoved:(WebEvent *)theEvent
{
    [self _forwardEvent:theEvent];
}

- (void)keyDown:(WebEvent *)event
{ 
    [self _forwardEvent:event];
}
- (void)keyUp:(WebEvent *)event
{ 
    [self _forwardEvent:event];
}

#if ENABLE(TOUCH_EVENTS)
- (void)touch:(WebEvent *)event
{
    [self _forwardEvent:event];
}
#endif

- (WAKResponder *)nextResponder { return nil; }

- (void)insertText:(NSString *)text
{
    UNUSED_PARAM(text);
}

- (void)deleteBackward:(id)sender
{
    UNUSED_PARAM(sender);
}

- (void)deleteForward:(id)sender
{
    UNUSED_PARAM(sender);
}

- (void)insertParagraphSeparator:(id)sender
{
    UNUSED_PARAM(sender);
}

- (void)moveDown:(id)sender
{
    UNUSED_PARAM(sender);
}

- (void)moveDownAndModifySelection:(id)sender
{
    UNUSED_PARAM(sender);
}

- (void)moveLeft:(id)sender
{
    UNUSED_PARAM(sender);
}

- (void)moveLeftAndModifySelection:(id)sender
{
    UNUSED_PARAM(sender);
}

- (void)moveRight:(id)sender
{
    UNUSED_PARAM(sender);
}

- (void)moveRightAndModifySelection:(id)sender
{
    UNUSED_PARAM(sender);
}

- (void)moveUp:(id)sender
{
    UNUSED_PARAM(sender);
}

- (void)moveUpAndModifySelection:(id)sender
{
    UNUSED_PARAM(sender);
}

- (void)mouseUp:(WebEvent *)event 
{
    [self _forwardEvent:event];
}

- (void)mouseDown:(WebEvent *)event 
{
    [self _forwardEvent:event];
}

- (BOOL)acceptsFirstResponder { return true; }
- (BOOL)becomeFirstResponder { return true; }
- (BOOL)resignFirstResponder { return true; }

- (BOOL)tryToPerform:(SEL)anAction with:(id)anObject 
{ 
    if ([self respondsToSelector:anAction]) {
        [self performSelector:anAction withObject:anObject];
        return YES;
    }
    return NO; 
}

@end

#endif // PLATFORM(IOS_FAMILY)
