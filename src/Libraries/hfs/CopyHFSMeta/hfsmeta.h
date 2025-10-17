/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 3, 2023.
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

#ifndef _HFS_META_H
# define _HFS_META_H

# include <libkern/OSByteOrder.h>
# include <hfs/hfs_format.h>

# include "Data.h"

/*
 * The in-core description of the volume.  We care about
 * the primary and alternate header locations, the alternate
 * and primary journal offset locations, the size of the device,
 * the two headers, and the two journal info blocks.
 */
struct VolumeDescriptor {
	off_t priOffset;	// Offset of primary volume header
	off_t altOffset;	// Offset of alternate volume header
	off_t priJInfoOffset;	// Offset of journal info block from primary header
	off_t altJInfoOffset;	// and for the alt.  May be the same as the above.
	off_t deviceSize;	// The size of the entire device
	JournalInfoBlock	priJournal;	// Primary journal block
	JournalInfoBlock	altJournal;	// Alternate journal, if different
	HFSPlusVolumeHeader priHeader;	// The primary header
	HFSPlusVolumeHeader altHeader;	// And the alternate
};
typedef struct VolumeDescriptor VolumeDescriptor_t;

/*
 * The input device description.
 */
struct DeviceInfo {
	char *devname;
	int fd;
	off_t size;
	int blockSize;
	off_t blockCount;
};
typedef struct DeviceInfo DeviceInfo_t;

# define S16(x)	OSSwapBigToHostInt16(x)
# define S32(x)	OSSwapBigToHostInt32(x)
# define S64(x)	OSSwapBigToHostInt64(x)

ssize_t GetBlock(DeviceInfo_t*, off_t, uint8_t*);
int ScanExtents(VolumeObjects_t *, int);

/*
 * The IOWrapper structure is used to do input and output on
 * the target -- which may be a device node, or a sparse bundle.
 * writer() is used to copy a particular amount of data from the source device;
 * reader() is used to get some data from the destination device (e.g., the header);
 * getprog() is used to find what the stored progress was (if any);
 * setprog() is used to write out the progress status so far.
 * cleanup() is called when the copy is done.
 */
struct IOWrapper {
	ssize_t (*writer)(struct IOWrapper *ctx,DeviceInfo_t *devp, off_t start, off_t len, void (^bp)(off_t));
	ssize_t (*reader)(struct IOWrapper *ctx, off_t start, void *buffer, off_t len);
	off_t (*getprog)(struct IOWrapper *ctx);
	void (*setprog)(struct IOWrapper *ctx, off_t prog);
	int (*cleanup)(struct IOWrapper *ctx);
	void *context;
};
typedef struct IOWrapper IOWrapper_t;

extern ssize_t UnalignedRead(DeviceInfo_t *, void *, size_t, off_t);

extern int debug, verbose, printProgress;

#endif /* _HFS_META_H */
