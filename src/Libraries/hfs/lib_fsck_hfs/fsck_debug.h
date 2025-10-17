/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 22, 2025.
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
#ifndef __FSCK_DEBUG__
#define __FSCK_DEBUG__

#include <sys/types.h>

enum debug_message_type {
	/* Type of information */
	d_info		=	0x0001,	/* Normal information messages during execution */
	d_error		= 	0x0002,	/* Error messages */

	/* Category of verify/repair operation */
	d_xattr		=	0x0010,	/* Extended attributes related messages */
	d_overlap	=	0x0020,	/* Overlap extents related messages */
	d_trim		=	0x0040,	/* TRIM (discard/unmap) related messages */
	
	d_dump_record = 0x0400,	/* Dump corrupt keys and records */
	d_dump_node	=	0x0800,	/* In hfs_swap_BTNode or BTCheck, dump out damaged nodes */
	d_check_slink	=	0x1000,	/* Read the contents of a symlink and check length */
};

void HexDump(const void *p_arg, unsigned length, int showOffsets);

#endif /* __FSCK_DEBUG__ */
