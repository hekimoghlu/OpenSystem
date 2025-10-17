/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 9, 2025.
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
#include <tk.h>
#include "ttkTheme.h"
#include "ttkWidget.h"

typedef struct {
    WidgetCore		*corePtr;	/* widget to track */
    Ttk_Layout		tracking;	/* current layout being tracked */
    Ttk_Element 	activeElement;	/* element under the mouse cursor */
    Ttk_Element 	pressedElement; /* currently pressed element */
} ElementStateTracker;

/*
 * ActivateElement(es, node) --
 * 	Make 'node' the active element if non-NULL.
 * 	Deactivates the currently active element if different.
 *
 * 	The active element has TTK_STATE_ACTIVE set _unless_
 * 	another element is 'pressed'
 */
static void ActivateElement(ElementStateTracker *es, Ttk_Element element)
{
    if (es->activeElement == element) {
	/* No change */
	return;
    }

    if (!es->pressedElement) {
	if (es->activeElement) {
	    /* Deactivate old element */
	    Ttk_ChangeElementState(es->activeElement, 0,TTK_STATE_ACTIVE);
	}
	if (element) {
	    /* Activate new element */
	    Ttk_ChangeElementState(element, TTK_STATE_ACTIVE,0);
	}
	TtkRedisplayWidget(es->corePtr);
    }

    es->activeElement = element;
}

/* ReleaseElement --
 * 	Releases the currently pressed element, if any.
 */
static void ReleaseElement(ElementStateTracker *es)
{
    if (!es->pressedElement)
	return;

    Ttk_ChangeElementState(
	es->pressedElement, 0,TTK_STATE_PRESSED|TTK_STATE_ACTIVE);
    es->pressedElement = 0;

    /* Reactivate element under the mouse cursor:
     */
    if (es->activeElement)
	Ttk_ChangeElementState(es->activeElement, TTK_STATE_ACTIVE,0);

    TtkRedisplayWidget(es->corePtr);
}

/* PressElement --
 * 	Presses the specified element.
 */
static void PressElement(ElementStateTracker *es, Ttk_Element element)
{
    if (es->pressedElement) {
	ReleaseElement(es);
    }

    if (element) {
	Ttk_ChangeElementState(
	    element, TTK_STATE_PRESSED|TTK_STATE_ACTIVE, 0);
    }

    es->pressedElement = element;
    TtkRedisplayWidget(es->corePtr);
}

/* ElementStateEventProc --
 * 	Event handler for tracking element states.
 */

static const unsigned ElementStateMask =
      ButtonPressMask
    | ButtonReleaseMask
    | PointerMotionMask
    | LeaveWindowMask
    | EnterWindowMask
    | StructureNotifyMask
    ;

static void
ElementStateEventProc(ClientData clientData, XEvent *ev)
{
    ElementStateTracker *es = clientData;
    Ttk_Layout layout = es->corePtr->layout;
    Ttk_Element element;

    /* Guard against dangling pointers [#2431428]
     */
    if (es->tracking != layout) {
	es->pressedElement = es->activeElement = 0;
	es->tracking = layout;
    }

    switch (ev->type)
    {
	case MotionNotify :
	    element = Ttk_IdentifyElement(
		layout, ev->xmotion.x, ev->xmotion.y);
	    ActivateElement(es, element);
	    break;
	case LeaveNotify:
	    ActivateElement(es, 0);
	    if (ev->xcrossing.mode == NotifyGrab)
		PressElement(es, 0);
	    break;
	case EnterNotify:
	    element = Ttk_IdentifyElement(
		layout, ev->xcrossing.x, ev->xcrossing.y);
	    ActivateElement(es, element);
	    break;
	case ButtonPress:
	    element = Ttk_IdentifyElement(
		layout, ev->xbutton.x, ev->xbutton.y);
	    if (element)
		PressElement(es, element);
	    break;
	case ButtonRelease:
	    ReleaseElement(es);
	    break;
	case DestroyNotify:
	    /* Unregister this event handler and free client data.
	     */
	    Tk_DeleteEventHandler(es->corePtr->tkwin,
		    ElementStateMask, ElementStateEventProc, es);
	    ckfree(clientData);
	    break;
    }
}

/*
 * TtkTrackElementState --
 * 	Register an event handler to manage the 'pressed'
 * 	and 'active' states of individual widget elements.
 */

void TtkTrackElementState(WidgetCore *corePtr)
{
    ElementStateTracker *es = (ElementStateTracker*)ckalloc(sizeof(*es));
    es->corePtr = corePtr;
    es->tracking = 0;
    es->activeElement = es->pressedElement = 0;
    Tk_CreateEventHandler(corePtr->tkwin,
	    ElementStateMask,ElementStateEventProc,es);
}

