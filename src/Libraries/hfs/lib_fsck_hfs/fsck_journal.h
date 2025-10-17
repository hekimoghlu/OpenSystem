/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 24, 2025.
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
#ifndef _FSCK_JOURNAL_H
#define _FSCK_JOURNAL_H

#include <sys/cdefs.h>

#include <sys/types.h>

/*
 * The guts of the journal:  a descriptor for which
 * block number on the data disk is to be written.
 */
typedef struct block_info {
	uint64_t	bnum;
	uint32_t	bsize;
	uint32_t	next;
} __attribute__((__packed__)) block_info;

/*
 * A "transaction," for want of a better word.
 * This contains a series of block_info, in the
 * binfo array, which are used to modify the
 * filesystem.
 */
typedef struct block_list_header {
	uint16_t	max_blocks;
	uint16_t	num_blocks;
	uint32_t	bytes_used;
	uint32_t	checksum;
	uint32_t	pad;
	block_info	binfo[1];
} __attribute__((__packed__)) block_list_header;

/*
 * This is written to block zero of the journal and it
 * maintains overall state about the journal.
 */
typedef struct journal_header {
    int32_t        magic;
    int32_t        endian;
    off_t	 start;         // zero-based byte offset of the start of the first transaction
    off_t	end;           // zero-based byte offset of where free space begins
    off_t          size;          // size in bytes of the entire journal
    int32_t        blhdr_size;    // size in bytes of each block_list_header in the journal
    int32_t        checksum;
    int32_t        jhdr_size;     // block size (in bytes) of the journal header
    uint32_t       sequence_num;  // NEW FIELD: a monotonically increasing value assigned to all txn's
} __attribute__((__packed__)) journal_header;

#define JOURNAL_HEADER_MAGIC	0x4a4e4c78   // 'JNLx'
#define	OLD_JOURNAL_HEADER_MAGIC	0x4a484452   // 'JHDR'
#define ENDIAN_MAGIC	0x12345678

//
// we only checksum the original size of the journal_header to remain
// backwards compatible.  the size of the original journal_header is
// everything up to the the sequence_num field, hence we use the
// offsetof macro to calculate the size.
//
#define JOURNAL_HEADER_CKSUM_SIZE  (offsetof(struct journal_header, sequence_num))

#define OLD_JOURNAL_HEADER_MAGIC  0x4a484452   // 'JHDR'

/*
 * The function used by fsck_hfs to replay the journal.
 * It's modeled on the kernel function.
 *
 * For the do_write_b block, the offset argument is in bytes --
 * the journal replay code will convert from journal block to
 * bytes.
 */

int	journal_open(int jdev,
		     off_t         offset,
		     off_t         journal_size,
		     size_t        min_fs_block_size,
		     uint32_t       flags,
		     const char	*jdev_name,
		     int (^do_write_b)(off_t, void *, size_t));

#endif /* !_FSCK_JOURNAL_H */
