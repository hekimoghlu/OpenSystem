/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 9, 2022.
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
/*      Copyright (c) 1992 NeXT Computer, Inc.  All rights reserved.
 *
 * kmreg_com.h - machine independent km ioctl interface.
 *
 * HISTORY
 * 16-Jan-92    Doug Mitchell at NeXT
 *      Created.
 */

#ifdef  KERNEL_PRIVATE

#ifndef _BSD_DEV_KMREG_COM_
#define _BSD_DEV_KMREG_COM_

#include <sys/types.h>
#include <sys/ioctl.h>

/*
 * Colors for fg, bg in struct km_drawrect
 */
#define KM_COLOR_WHITE          0
#define KM_COLOR_LTGRAY         1
#define KM_COLOR_DKGRAY         2
#define KM_COLOR_BLACK          3

/*
 * The data to be rendered is treated as a pixmap of 2 bit pixels.
 * The most significant bits of each byte is the leftmost pixel in that
 * byte.  Pixel values are assigned as described above.
 *
 * Each scanline should start on a 4 pixel boundry within the bitmap,
 * and should be a multiple of 4 pixels in length.
 *
 * For the KMIOCERASERECT call, 'data' should be an integer set to the
 * color to be used for the clear operation (data.fill).
 * A rect at (x,y) measuring 'width' by 'height' will be cleared to
 * the specified value.
 */
struct km_drawrect {
	unsigned short x;       /* Upper left corner of rect to be imaged. */
	unsigned short y;
	unsigned short width;   /* Width and height of rect to be imaged,
	                         * in pixels */
	unsigned short height;
	union {
		void *bits;     /* Pointer to 2 bit per pixel raster data. */
		int   fill;     /* Const color for erase operation. */
	} data;
};

/*
 * Argument to KMIOCANIMCTL.
 */
typedef enum {
	KM_ANIM_STOP,           /* stop permanently */
	KM_ANIM_SUSPEND,        /* suspend */
	KM_ANIM_RESUME          /* resume */
} km_anim_ctl_t;

#define KMIOCPOPUP      _IO('k', 1)             /* popup new window */
#define KMIOCRESTORE    _IO('k', 2)             /* restore background */
#define KMIOCDUMPLOG    _IO('k', 3)             /* dump message log */
#define KMIOCDRAWRECT   _IOW('k', 5, struct km_drawrect)  /* Draw rect from
	                                                   * bits */
#define KMIOCERASERECT  _IOW('k', 6, struct km_drawrect)  /* Erase a rect */

#ifdef  KERNEL_PRIVATE
#define KMIOCDISABLCONS _IO('k', 8)             /* disable console messages */
#endif  /* KERNEL_PRIVATE */

#define KMIOCANIMCTL    _IOW('k',9, km_anim_ctl_t)
/* stop animation */
#define KMIOCSTATUS     _IOR('k',10, int)       /* get status bits */
#define KMIOCSIZE       _IOR('k',11, struct winsize) /* get screen size */

/*
 * Status bits returned via KMIOCSTATUS.
 */
#define KMS_SEE_MSGS    0x00000001

#endif  /* _BSD_DEV_KMREG_COM_ */

#endif  /* KERNEL_PRIVATE */
