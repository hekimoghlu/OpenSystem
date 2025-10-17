/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 20, 2024.
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
#ifndef _TK3D
#define _TK3D

#include <tkInt.h>

#ifdef BUILD_tk
# undef TCL_STORAGE_CLASS
# define TCL_STORAGE_CLASS DLLEXPORT
#endif

/*
 * One of the following data structures is allocated for each 3-D border
 * currently in use. Structures of this type are indexed by borderTable, so
 * that a single structure can be shared for several uses.
 */

typedef struct TkBorder {
    Screen *screen;		/* Screen on which the border will be used. */
    Visual *visual;		/* Visual for all windows and pixmaps using
				 * the border. */
    int depth;			/* Number of bits per pixel of drawables where
				 * the border will be used. */
    Colormap colormap;		/* Colormap out of which pixels are
				 * allocated. */
    int resourceRefCount;	/* Number of active uses of this color (each
				 * active use corresponds to a call to
				 * Tk_Alloc3DBorderFromObj or Tk_Get3DBorder).
				 * If this count is 0, then this structure is
				 * no longer valid and it isn't present in
				 * borderTable: it is being kept around only
				 * because there are objects referring to it.
				 * The structure is freed when
				 * resourceRefCount and objRefCount are both
				 * 0. */
    int objRefCount;		/* The number of Tcl objects that reference
				 * this structure. */
    XColor *bgColorPtr;		/* Background color (intensity between
				 * lightColorPtr and darkColorPtr). */
    XColor *darkColorPtr;	/* Color for darker areas (must free when
				 * deleting structure). NULL means shadows
				 * haven't been allocated yet.*/
    XColor *lightColorPtr;	/* Color used for lighter areas of border
				 * (must free this when deleting structure).
				 * NULL means shadows haven't been allocated
				 * yet. */
    Pixmap shadow;		/* Stipple pattern to use for drawing shadows
				 * areas. Used for displays with <= 64 colors
				 * or where colormap has filled up. */
    GC bgGC;			/* Used (if necessary) to draw areas in the
				 * background color. */
    GC darkGC;			/* Used to draw darker parts of the border.
				 * None means the shadow colors haven't been
				 * allocated yet.*/
    GC lightGC;			/* Used to draw lighter parts of the border.
				 * None means the shadow colors haven't been
				 * allocated yet. */
    Tcl_HashEntry *hashPtr;	/* Entry in borderTable (needed in order to
				 * delete structure). */
    struct TkBorder *nextPtr;	/* Points to the next TkBorder structure with
				 * the same color name. Borders with the same
				 * name but different screens or colormaps are
				 * chained together off a single entry in
				 * borderTable. */
} TkBorder;

/*
 * Maximum intensity for a color:
 */

#define MAX_INTENSITY 65535

/*
 * Declarations for platform specific interfaces used by this module.
 */

MODULE_SCOPE TkBorder	*TkpGetBorder(void);
MODULE_SCOPE void	TkpGetShadows(TkBorder *borderPtr, Tk_Window tkwin);
MODULE_SCOPE void	TkpFreeBorder(TkBorder *borderPtr);

# undef TCL_STORAGE_CLASS
# define TCL_STORAGE_CLASS DLLIMPORT

#endif /* _TK3D */
