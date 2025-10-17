/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 22, 2025.
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
#include "tkInt.h"

/*
 *----------------------------------------------------------------------
 *
 * XCreateBitmapFromData --
 *
 *	Construct a single plane pixmap from bitmap data.
 *
 *	NOTE: This procedure has the correct behavior on Windows and the
 *	Macintosh, but not on UNIX. This is probably because the emulation for
 *	XPutImage on those platforms compensates for whatever is wrong here
 *	:-)
 *
 * Results:
 *	Returns a new Pixmap.
 *
 * Side effects:
 *	Allocates a new bitmap and drawable.
 *
 *----------------------------------------------------------------------
 */

Pixmap
XCreateBitmapFromData(
    Display *display,
    Drawable d,
    _Xconst char *data,
    unsigned int width,
    unsigned int height)
{
    XImage *ximage;
    GC gc;
    Pixmap pix;

    pix = Tk_GetPixmap(display, d, (int) width, (int) height, 1);
    gc = XCreateGC(display, pix, 0, NULL);
    if (gc == NULL) {
	return None;
    }
    ximage = XCreateImage(display, NULL, 1, XYBitmap, 0, (char*) data, width,
	    height, 8, (width + 7) / 8);
    ximage->bitmap_bit_order = LSBFirst;
    _XInitImageFuncPtrs(ximage);
    TkPutImage(NULL, 0, display, pix, gc, ximage, 0, 0, 0, 0, width, height);
    ximage->data = NULL;
    XDestroyImage(ximage);
    XFreeGC(display, gc);
    return pix;
}

/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 4
 * fill-column: 78
 * End:
 */
