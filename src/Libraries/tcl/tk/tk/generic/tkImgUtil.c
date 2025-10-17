/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 16, 2022.
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
#include "xbytes.h"


/*
 *----------------------------------------------------------------------
 *
 * TkAlignImageData --
 *
 *	This function takes an image and copies the data into an aligned
 *	buffer, performing any necessary bit swapping.
 *
 * Results:
 *	Returns a newly allocated buffer that should be freed by the caller.
 *
 * Side effects:
 *	None.
 *
 *----------------------------------------------------------------------
 */

char *
TkAlignImageData(
    XImage *image,		/* Image to be aligned. */
    int alignment,		/* Number of bytes to which the data should be
				 * aligned (e.g. 2 or 4) */
    int bitOrder)		/* Desired bit order: LSBFirst or MSBFirst. */
{
    long dataWidth;
    char *data, *srcPtr, *destPtr;
    int i, j;

    if (image->bits_per_pixel != 1) {
	Tcl_Panic(
		"TkAlignImageData: Can't handle image depths greater than 1.");
    }

    /*
     * Compute line width for output data buffer.
     */

    dataWidth = image->bytes_per_line;
    if (dataWidth % alignment) {
	dataWidth += (alignment - (dataWidth % alignment));
    }

    data = ckalloc((unsigned) dataWidth * image->height);

    destPtr = data;
    for (i = 0; i < image->height; i++) {
	srcPtr = &image->data[i * image->bytes_per_line];
	for (j = 0; j < dataWidth; j++) {
	    if (j >= image->bytes_per_line) {
		*destPtr = 0;
	    } else if (image->bitmap_bit_order != bitOrder) {
		*destPtr = xBitReverseTable[(unsigned char)(*(srcPtr++))];
	    } else {
		*destPtr = *(srcPtr++);
	    }
	    destPtr++;
	}
    }
    return data;
}

/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 4
 * fill-column: 78
 * End:
 */
