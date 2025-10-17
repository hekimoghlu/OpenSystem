/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 15, 2023.
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
#ifndef _TKCOLOR
#define _TKCOLOR

#include <tkInt.h>

#ifdef BUILD_tk
#undef TCL_STORAGE_CLASS
#define TCL_STORAGE_CLASS DLLEXPORT
#endif

/*
 * One of the following data structures is used to keep track of each color
 * that is being used by the application; typically there is a colormap entry
 * allocated for each of these colors.
 */

#define TK_COLOR_BY_NAME	1
#define TK_COLOR_BY_VALUE	2

#define COLOR_MAGIC ((unsigned int) 0x46140277)

typedef struct TkColor {
    XColor color;		/* Information about this color. */
    unsigned int magic;		/* Used for quick integrity check on this
				 * structure. Must always have the value
				 * COLOR_MAGIC. */
    GC gc;			/* Simple gc with this color as foreground
				 * color and all other fields defaulted. May
				 * be None. */
    Screen *screen;		/* Screen where this color is valid. Used to
				 * delete it, and to find its display. */
    Colormap colormap;		/* Colormap from which this entry was
				 * allocated. */
    Visual *visual;		/* Visual associated with colormap. */
    int resourceRefCount;	/* Number of active uses of this color (each
				 * active use corresponds to a call to
				 * Tk_AllocColorFromObj or Tk_GetColor). If
				 * this count is 0, then this TkColor
				 * structure is no longer valid and it isn't
				 * present in a hash table: it is being kept
				 * around only because there are objects
				 * referring to it. The structure is freed
				 * when resourceRefCount and objRefCount are
				 * both 0. */
    int objRefCount;		/* The number of Tcl objects that reference
				 * this structure. */
    int type;			/* TK_COLOR_BY_NAME or TK_COLOR_BY_VALUE. */
    Tcl_HashEntry *hashPtr;	/* Pointer to hash table entry for this
				 * structure. (for use in deleting entry). */
    struct TkColor *nextPtr;	/* Points to the next TkColor structure with
				 * the same color name. Colors with the same
				 * name but different screens or colormaps are
				 * chained together off a single entry in
				 * nameTable. For colors in valueTable (those
				 * allocated by Tk_GetColorByValue) this field
				 * is always NULL. */
} TkColor;

/*
 * Common APIs exported from all platform-specific implementations.
 */

#ifndef TkpFreeColor
MODULE_SCOPE void	TkpFreeColor(TkColor *tkColPtr);
#endif
MODULE_SCOPE TkColor *	TkpGetColor(Tk_Window tkwin, Tk_Uid name);
MODULE_SCOPE TkColor *	TkpGetColorByValue(Tk_Window tkwin, XColor *colorPtr);

#undef TCL_STORAGE_CLASS
#define TCL_STORAGE_CLASS DLLIMPORT

#endif /* _TKCOLOR */
