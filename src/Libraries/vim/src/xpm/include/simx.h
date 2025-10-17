/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 15, 2022.
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
/*****************************************************************************\
* simx.h: 0.1a                                                                *
*                                                                             *
* This emulates some Xlib functionality for MSW. It's not a general solution, *
* it is close related to XPM-lib. It is only intended to satisfy what is need *
* there. Thus allowing to read XPM files under MS windows.                    *
*                                                                             *
* Developed by HeDu 3/94 (hedu@cul-ipn.uni-kiel.de)                           *
\*****************************************************************************/


#ifndef _SIMX_H
#define _SIMX_H

#ifdef FOR_MSW

#include "windows.h"			/* MS windows GDI types */

/*
 * minimal portability layer between ansi and KR C
 */
/* this comes from xpm.h, and is here again, to avoid complicated
    includes, since this is included from xpm.h */
/* these defines get undefed at the end of this file */
#if __STDC__ || defined(__cplusplus) || defined(c_plusplus)
 /* ANSI || C++ */
#define FUNC(f, t, p) extern t f p
#define LFUNC(f, t, p) static t f p
#else /* k&R */
#define FUNC(f, t, p) extern t f()
#define LFUNC(f, t, p) static t f()
#endif


FUNC(boundCheckingMalloc, void *, (long s));
FUNC(boundCheckingCalloc, void *, (long num, long s));
FUNC(boundCheckingRealloc, void *, (void *p, long s));

/* define MSW types for X window types,
   I don't know much about MSW, but the following defines do the job */

typedef HDC Display;			/* this should be similar */
typedef void *Screen;			/* not used */
typedef void *Visual;			/* not used yet, is for GRAY, COLOR,
					 * MONO */

typedef void *Colormap;			/* should be COLORPALETTE, not done
					 * yet */

typedef COLORREF Pixel;

#define PIXEL_ALREADY_TYPEDEFED		/* to let xpm.h know about it */

typedef struct {
    Pixel pixel;
    BYTE red, green, blue;
}      XColor;

typedef struct {
    HBITMAP bitmap;
    unsigned int width;
    unsigned int height;
    unsigned int depth;
}      XImage;

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif
/* some replacements for X... functions */

/* XDefaultXXX */
    FUNC(XDefaultVisual, Visual *, (Display *display, Screen *screen));
    FUNC(XDefaultScreen, Screen *, (Display *d));
    FUNC(XDefaultColormap, Colormap *, (Display *display, Screen *screen));
    FUNC(XDefaultDepth, int, (Display *d, Screen *s));

/* color related */
    FUNC(XParseColor, int, (Display *, Colormap *, char *, XColor *));
    FUNC(XAllocColor, int, (Display *, Colormap *, XColor *));
    FUNC(XQueryColors, void, (Display *display, Colormap *colormap,
			      XColor *xcolors, int ncolors));
    FUNC(XFreeColors, int, (Display *d, Colormap cmap,
			    unsigned long pixels[],
			    int npixels, unsigned long planes));
/* XImage */
    FUNC(XCreateImage, XImage *, (Display *, Visual *, int depth, int format,
				  int x, int y, int width, int height,
				  int pad, int foo));

/* free and destroy bitmap */
    FUNC(XDestroyImage, void /* ? */ , (XImage *));
/* free only, bitmap remains */
    FUNC(XImageFree, void, (XImage *));
#if defined(__cplusplus) || defined(c_plusplus)
} /* end of extern "C" */
#endif /* cplusplus */

#define ZPixmap 1			/* not really used */
#define XYBitmap 1			/* not really used */

#ifndef True
#define True 1
#define False 0
#endif
#ifndef Bool
typedef BOOL Bool;		/* take MSW bool */
#endif
/* make these local here, simx.c gets the same from xpm.h */
#undef LFUNC
#undef FUNC

#endif /* def FOR_MSW */

#endif /* _SIMX_H */
