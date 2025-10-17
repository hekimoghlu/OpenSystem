/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 8, 2025.
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
#include "QuickTimeTclWin.h"
#include "QuickTimeTcl.h"

/*
 *----------------------------------------------------------------------
 *
 * TkSetMacColor --
 *
 *	Populates a Macintosh RGBColor structure from a X style
 *	pixel value.
 *
 * Results:
 *	Returns false if not a real pixel, true otherwise.
 *
 * Side effects:
 *	The variable macColor is updated to the pixels value.
 *
 *----------------------------------------------------------------------
 */

int
TkSetMacColor(
    unsigned long pixel,	/* Pixel value to convert. */
    RGBColor *macColor)		/* Mac color struct to modify. */
{
#ifdef WORDS_BIGENDIAN
    macColor->blue = (unsigned short) ((pixel & 0xFF) << 8);
    macColor->green = (unsigned short) (((pixel >> 8) & 0xFF) << 8);
    macColor->red = (unsigned short) (((pixel >> 16) & 0xFF) << 8);
#else
    macColor->red = (unsigned short) (((pixel >> 24) & 0xFF) << 8);
    macColor->green = (unsigned short) (((pixel >> 16) & 0xFF) << 8);
    macColor->blue = (unsigned short) (((pixel >> 8) & 0xFF) << 8);
#endif
    return true;
}
