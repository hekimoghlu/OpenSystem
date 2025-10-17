/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 5, 2022.
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
/*
 * This program is used to determine the screen dimensions on OS/2 systems.
 * Adapted from code written by Kyosuke Tokoro (NBG01720@nifty.ne.jp).
 */

/*
 * When I wrote this routine, I consulted some part of the source code 
 * of the xwininfo utility by X Consortium.
 *
 * Copyright (c) 1987, X Consortium
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * X CONSORTIUM BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
 * AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * Except as contained in this notice, the name of the X Consortium shall not
 * be used in advertising or otherwise to promote the sale, use or other
 * dealings in this Software without prior written authorization from the X
 * Consortium.
 */
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <stdlib.h>
#include <stdio.h>

static int get_winsize(dpy, window, p_width, p_height)
	Display *dpy;
	Window window;
	int *p_width;
	int *p_height;
{
	XWindowAttributes win_attributes;
	XSizeHints hints;
	long longjunk;

	if (!XGetWindowAttributes(dpy, window, &win_attributes))
		return 1;
	if (!XGetWMNormalHints(dpy, window, &hints, &longjunk))
		return 1;
	if (!(hints.flags & PResizeInc))
		return 1;
	if (hints.width_inc == 0 || hints.height_inc == 0)
		return 1;
	if (!(hints.flags & (PBaseSize|PMinSize)))
		return 1;
	if (hints.flags & PBaseSize)
	{
		win_attributes.width -= hints.base_width;
		win_attributes.height -= hints.base_height;
	} else
	{
		win_attributes.width -= hints.min_width;
		win_attributes.height -= hints.min_height;
	}
	*p_width = win_attributes.width / hints.width_inc;
	*p_height = win_attributes.height / hints.height_inc;
	return 0;
}

int main(argc, argv)
	int argc;
	char *argv[];
{
	char *cp;
	Display *dpy;
	int size[2];

	_scrsize(size);
	cp = getenv("WINDOWID");
	if (cp != NULL)
	{
		dpy = XOpenDisplay(NULL);
		if (dpy != NULL)
		{
			get_winsize(dpy, (Window) atol(cp), &size[0], &size[1]);
			XCloseDisplay(dpy);
		}
	}
	printf("%i %i\n", size[0], size[1]);
	return (0);
}
