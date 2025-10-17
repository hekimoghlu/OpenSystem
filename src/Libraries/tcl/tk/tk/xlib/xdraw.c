/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 25, 2023.
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
#include "tk.h"

/*
 *----------------------------------------------------------------------
 *
 * XDrawLine --
 *
 *	Draw a single line between two points in a given drawable.
 *
 * Results:
 *	None.
 *
 * Side effects:
 *	Draws a single line segment.
 *
 *----------------------------------------------------------------------
 */

void
XDrawLine(
    Display *display,
    Drawable d,
    GC gc,
    int x1, int y1,
    int x2, int y2)		/* Coordinates of line segment. */
{
    XPoint points[2];

    points[0].x = x1;
    points[0].y = y1;
    points[1].x = x2;
    points[1].y = y2;
    XDrawLines(display, d, gc, points, 2, CoordModeOrigin);
}

/*
 *----------------------------------------------------------------------
 *
 * XFillRectangle --
 *
 *	Fills a rectangular area in the given drawable. This procedure is
 *	implemented as a call to XFillRectangles.
 *
 * Results:
 *	None
 *
 * Side effects:
 *	Fills the specified rectangle.
 *
 *----------------------------------------------------------------------
 */

void
XFillRectangle(
    Display *display,
    Drawable d,
    GC gc,
    int x,
    int y,
    unsigned int width,
    unsigned int height)
{
    XRectangle rectangle;
    rectangle.x = x;
    rectangle.y = y;
    rectangle.width = width;
    rectangle.height = height;
    XFillRectangles(display, d, gc, &rectangle, 1);
}

/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 4
 * fill-column: 78
 * End:
 */
