/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 9, 2022.
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
#ifndef _OSX_NTFS_COMPRESS_H
#define _OSX_NTFS_COMPRESS_H

#include <sys/errno.h>

#include <mach/memory_object_types.h>

#include "ntfs_inode.h"
#include "ntfs_types.h"

__private_extern__ errno_t ntfs_read_compressed(ntfs_inode *ni,
		ntfs_inode *raw_ni, s64 ofs, const int start_count,
		u8 *dst_start, upl_page_info_t *pl, int ioflags);

#endif /* !_OSX_NTFS_COMPRESS_H */
