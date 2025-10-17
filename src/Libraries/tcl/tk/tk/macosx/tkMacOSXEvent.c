/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 26, 2022.
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
#include "tkMacOSXPrivate.h"
#include "tkMacOSXEvent.h"
#include "tkMacOSXDebug.h"

#pragma mark TKApplication(TKEvent)

enum {
    NSWindowWillMoveEventType = 20
};

@implementation TKApplication(TKEvent)
/* TODO: replace by +[addLocalMonitorForEventsMatchingMask ? */
- (NSEvent *)tkProcessEvent:(NSEvent *)theEvent {
#ifdef TK_MAC_DEBUG_EVENTS
    TKLog(@"-[%@(%p) %s] %@", [self class], self, _cmd, theEvent);
#endif
    NSEvent	    *processedEvent = theEvent;
    NSEventType	    type = [theEvent type];
    NSInteger	    subtype;
    NSUInteger	    flags;

    switch ((NSInteger)type) {
    case NSAppKitDefined:
        subtype = [theEvent subtype];

	switch (subtype) {
	case NSApplicationActivatedEventType:
	    break;
	case NSApplicationDeactivatedEventType:
	    break;
	case NSWindowExposedEventType:
	case NSScreenChangedEventType:
	    break;
	case NSWindowMovedEventType:
	    break;
        case NSWindowWillMoveEventType:
            break;

        default:
            break;
	}
	break;
    case NSKeyUp:
    case NSKeyDown:
    case NSFlagsChanged:
	flags = [theEvent modifierFlags];
	processedEvent = [self tkProcessKeyEvent:theEvent];
	break;
    case NSLeftMouseDown:
    case NSLeftMouseUp:
    case NSRightMouseDown:
    case NSRightMouseUp:
    case NSLeftMouseDragged:
    case NSRightMouseDragged:
    case NSMouseMoved:
    case NSMouseEntered:
    case NSMouseExited:
    case NSScrollWheel:
    case NSOtherMouseDown:
    case NSOtherMouseUp:
    case NSOtherMouseDragged:
    case NSTabletPoint:
    case NSTabletProximity:
	processedEvent = [self tkProcessMouseEvent:theEvent];
	break;
#if 0
    case NSSystemDefined:
        subtype = [theEvent subtype];
	break;
    case NSApplicationDefined: {
	id win;
	win = [theEvent window];
	break;
	}
    case NSCursorUpdate:
        break;
#if MAC_OS_X_VERSION_MAX_ALLOWED >= 1060
    case NSEventTypeGesture:
    case NSEventTypeMagnify:
    case NSEventTypeRotate:
    case NSEventTypeSwipe:
    case NSEventTypeBeginGesture:
    case NSEventTypeEndGesture:
        break;
#endif
#endif

    default:
	break;
    }
    return processedEvent;
}
@end

#pragma mark -

/*
 *----------------------------------------------------------------------
 *
 * TkMacOSXFlushWindows --
 *
 *	This routine flushes all the windows of the application. It is
 *	called by XSync().
 *
 * Results:
 *	None.
 *
 * Side effects:
 *	Flushes all Carbon windows
 *
 *----------------------------------------------------------------------
 */

MODULE_SCOPE void
TkMacOSXFlushWindows(void)
{
    NSInteger windowCount;
    NSInteger *windowNumbers;

    NSCountWindows(&windowCount);
    if(windowCount) {
	windowNumbers = (NSInteger *) ckalloc(windowCount * sizeof(NSInteger));
	NSWindowList(windowCount, windowNumbers);
	for (NSInteger index = 0; index < windowCount; index++) {
	    NSWindow *w = [NSApp windowWithWindowNumber:windowNumbers[index]];
	    if (TkMacOSXGetXWindow(w)) {
		[w flushWindow];
	    }
	}
	ckfree((char*) windowNumbers);
    }
}

/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 4
 * fill-column: 79
 * coding: utf-8
 * End:
 */
