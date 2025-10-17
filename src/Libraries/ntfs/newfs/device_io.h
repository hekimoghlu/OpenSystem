/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 18, 2025.
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
#ifndef _NTFS_DEVICE_IO_H
#define _NTFS_DEVICE_IO_H

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifndef NO_NTFS_DEVICE_DEFAULT_IO_OPS

#ifndef __CYGWIN32__

/* Not on Cygwin; use standard Unix style low level device operations. */
#define ntfs_device_default_io_ops ntfs_device_unix_io_ops

#else /* __CYGWIN32__ */

#ifndef HDIO_GETGEO
#	define HDIO_GETGEO	0x301
/**
 * struct hd_geometry -
 */
struct hd_geometry {
	unsigned char heads;
	unsigned char sectors;
	unsigned short cylinders;
	unsigned long start;
};
#endif
#ifndef BLKGETSIZE
#	define BLKGETSIZE	0x1260
#endif
#ifndef BLKSSZGET
#	define BLKSSZGET	0x1268
#endif
#ifndef BLKGETSIZE64
#	define BLKGETSIZE64	0x80041272
#endif
#ifndef BLKBSZSET
#	define BLKBSZSET	0x40041271
#endif

/* On Cygwin; use Win32 low level device operations. */
#define ntfs_device_default_io_ops ntfs_device_win32_io_ops

#endif /* __CYGWIN32__ */


/* Forward declaration. */
struct ntfs_device_operations;

extern struct ntfs_device_operations ntfs_device_default_io_ops;

#endif /* NO_NTFS_DEVICE_DEFAULT_IO_OPS */

#endif /* defined _NTFS_DEVICE_IO_H */
