/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 24, 2025.
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
#include "tkWinInt.h"


/*
 *----------------------------------------------------------------------
 *
 * TkCreateRegion --
 *
 *	Construct an empty region.
 *
 * Results:
 *	Returns a new region handle.
 *
 * Side effects:
 *	None.
 *
 *----------------------------------------------------------------------
 */

TkRegion
TkCreateRegion(void)
{
    RECT rect;
    memset(&rect, 0, sizeof(RECT));
    return (TkRegion) CreateRectRgnIndirect(&rect);
}

/*
 *----------------------------------------------------------------------
 *
 * TkDestroyRegion --
 *
 *	Destroy the specified region.
 *
 * Results:
 *	None.
 *
 * Side effects:
 *	Frees the storage associated with the specified region.
 *
 *----------------------------------------------------------------------
 */

void
TkDestroyRegion(
    TkRegion r)
{
    DeleteObject((HRGN) r);
}

/*
 *----------------------------------------------------------------------
 *
 * TkClipBox --
 *
 *	Computes the bounding box of a region.
 *
 * Results:
 *	Sets rect_return to the bounding box of the region.
 *
 * Side effects:
 *	None.
 *
 *----------------------------------------------------------------------
 */

void
TkClipBox(
    TkRegion r,
    XRectangle* rect_return)
{
    RECT rect;

    GetRgnBox((HRGN)r, &rect);
    rect_return->x = (short) rect.left;
    rect_return->y = (short) rect.top;
    rect_return->width = (short) (rect.right - rect.left);
    rect_return->height = (short) (rect.bottom - rect.top);
}

/*
 *----------------------------------------------------------------------
 *
 * TkIntersectRegion --
 *
 *	Compute the intersection of two regions.
 *
 * Results:
 *	Returns the result in the dr_return region.
 *
 * Side effects:
 *	None.
 *
 *----------------------------------------------------------------------
 */

void
TkIntersectRegion(
    TkRegion sra,
    TkRegion srb,
    TkRegion dr_return)
{
    CombineRgn((HRGN) dr_return, (HRGN) sra, (HRGN) srb, RGN_AND);
}

/*
 *----------------------------------------------------------------------
 *
 * TkUnionRectWithRegion --
 *
 *	Create the union of a source region and a rectangle.
 *
 * Results:
 *	Returns the result in the dr_return region.
 *
 * Side effects:
 *	None.
 *
 *----------------------------------------------------------------------
 */

void
TkUnionRectWithRegion(
    XRectangle *rectangle,
    TkRegion src_region,
    TkRegion dest_region_return)
{
    HRGN rectRgn = CreateRectRgn(rectangle->x, rectangle->y,
	    rectangle->x + rectangle->width, rectangle->y + rectangle->height);

    CombineRgn((HRGN) dest_region_return, (HRGN) src_region,
	    (HRGN) rectRgn, RGN_OR);
    DeleteObject(rectRgn);
}

/*
 *----------------------------------------------------------------------
 *
 * TkpBuildRegionFromAlphaData --
 *
 *	Set up a rectangle of the given region based on the supplied alpha
 *	data.
 *
 * Results:
 *	None
 *
 * Side effects:
 *	The region is updated, with extra pixels added to it.
 *
 *----------------------------------------------------------------------
 */

void
TkpBuildRegionFromAlphaData(
    TkRegion region,
    unsigned int x, unsigned int y,
				/* Where in region to update. */
    unsigned int width, unsigned int height,
				/* Size of rectangle to update. */
    unsigned char *dataPtr,	/* Data to read from. */
    unsigned int pixelStride,	/* Num bytes from one piece of alpha data to
				 * the next in the line. */
    unsigned int lineStride)	/* Num bytes from one line of alpha data to
				 * the next line. */
{
    unsigned char *lineDataPtr;
    unsigned int x1, y1, end;
    HRGN rectRgn = CreateRectRgn(0,0,1,1); /* Workspace region. */

    for (y1 = 0; y1 < height; y1++) {
	lineDataPtr = dataPtr;
	for (x1 = 0; x1 < width; x1 = end) {
	    /*
	     * Search for first non-transparent pixel.
	     */

	    while ((x1 < width) && !*lineDataPtr) {
		x1++;
		lineDataPtr += pixelStride;
	    }
	    end = x1;

	    /*
	     * Search for first transparent pixel.
	     */

	    while ((end < width) && *lineDataPtr) {
		end++;
		lineDataPtr += pixelStride;
	    }
	    if (end > x1) {
		/*
		 * Manipulate Win32 regions directly; it's more efficient.
		 */

		SetRectRgn(rectRgn, (int) (x+x1), (int) (y+y1),
			(int) (x+end), (int) (y+y1+1));
		CombineRgn((HRGN) region, (HRGN) region, rectRgn, RGN_OR);
	    }
	}
	dataPtr += lineStride;
    }

    DeleteObject(rectRgn);
}

/*
 *----------------------------------------------------------------------
 *
 * TkRectInRegion --
 *
 *	Test whether a given rectangle overlaps with a region.
 *
 * Results:
 *	Returns RectanglePart or RectangleOut. Note that this is not a
 *	complete implementation since it doesn't test for RectangleIn.
 *
 * Side effects:
 *	None.
 *
 *----------------------------------------------------------------------
 */

int
TkRectInRegion(
    TkRegion r,			/* Region to inspect */
    int x, int y,		/* Top-left of rectangle */
    unsigned int width,		/* Width of rectangle */
    unsigned int height)	/* Height of rectangle */
{
    RECT rect;
    rect.top = y;
    rect.left = x;
    rect.bottom = y+height;
    rect.right = x+width;
    return RectInRegion((HRGN)r, &rect) ? RectanglePart : RectangleOut;
}

/*
 *----------------------------------------------------------------------
 *
 * TkSubtractRegion --
 *
 *	Compute the set-difference of two regions.
 *
 * Results:
 *	Returns the result in the dr_return region.
 *
 * Side effects:
 *	None.
 *
 *----------------------------------------------------------------------
 */

void
TkSubtractRegion(
    TkRegion sra,
    TkRegion srb,
    TkRegion dr_return)
{
    CombineRgn((HRGN) dr_return, (HRGN) sra, (HRGN) srb, RGN_DIFF);
}

/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 4
 * fill-column: 78
 * End:
 */
