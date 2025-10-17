/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 17, 2024.
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
#ifndef hfs_hfs_extents_test_h
#define hfs_hfs_extents_test_h

// Stuff that allow us to build hfs_extents.c for testing

#define KERNEL 1
#define __APPLE_API_PRIVATE 1
#define HFS_EXTENTS_TEST 1

#include <stdint.h>
#include <sys/errno.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <sys/param.h>
#include <assert.h>
#include <unistd.h>
#include <sys/queue.h>

#include "../core/hfs_format.h"

#define VTOF(v)				(v)->ffork
#define VTOHFS(v)			(v)->mount
#define VNODE_IS_RSRC(v)	(v)->is_rsrc
#define VTOC(v)				(v)->cnode

struct BTreeHint{
	unsigned long			writeCount;
	u_int32_t				nodeNum;			// node the key was last seen in
	u_int16_t				index;				// index then key was last seen at
	u_int16_t				reserved1;
	u_int32_t				reserved2;
};
typedef struct BTreeHint BTreeHint;

typedef struct BTreeIterator {
	BTreeHint	hint;
	uint16_t	version;
	uint16_t	reserved;
	uint32_t	hitCount;			// Total number of leaf records hit
	uint32_t	maxLeafRecs;		// Max leaf records over iteration
	struct extent_group *group;
	HFSPlusExtentKey key;
} BTreeIterator;

typedef struct filefork {
	uint32_t ff_blocks, ff_unallocblocks;
	LIST_HEAD(extent_groups, extent_group) groups;
} filefork_t;

typedef struct cnode {
	uint32_t c_fileid;
	filefork_t *c_datafork;
} cnode_t;

typedef struct hfsmount {
	cnode_t *hfs_extents_cp;
	uint32_t blockSize;
} hfsmount_t;

typedef struct vnode {
	filefork_t *ffork;
	hfsmount_t *mount;
	bool is_rsrc;
	cnode_t *cnode;
} *vnode_t;

struct FSBufferDescriptor {
	void *		bufferAddress;
	uint32_t	itemSize;
	uint32_t	itemCount;
};
typedef struct FSBufferDescriptor FSBufferDescriptor;

static inline int32_t
BTSearchRecord		(__unused filefork_t					*filePtr,
					 __unused BTreeIterator				*searchIterator,
					 __unused FSBufferDescriptor			*record,
					 __unused u_int16_t					*recordLen,
					 __unused BTreeIterator				*resultIterator )
{
	return ENOTSUP;
}

/* Constants for HFS fork types */
enum {
	kHFSDataForkType = 0x0, 	/* data fork */
	kHFSResourceForkType = 0xff	/* resource fork */
};

static inline void *hfs_malloc(size_t size)
{
    return malloc(size);
}

static inline void hfs_free(void *ptr, __unused size_t size)
{
    return free(ptr);
}

#define _hfs_new(type, count)                                  \
({                                                             \
        void *_ptr = NULL;                                     \
        typeof(count) _count = count;                          \
        size_t _size = sizeof(type) * _count;                  \
        _ptr = calloc(1,_size);                                \
        _ptr;                                                  \
})

#define _hfs_delete(ptr, type, count)                          \
({                                                             \
        typeof(ptr) _ptr = ptr;                                \
        __unused typeof(count) _count = count;                 \
        if (_ptr) {                                            \
                free(_ptr);                                    \
        }                                                      \
})

#define hfs_new(type, count) _hfs_new(type, count)
#define hfs_new_data(type, count) _hfs_new(type, count)
#define hfs_delete(ptr, type, count) _hfs_delete(ptr, type, count)
#define hfs_delete_data(ptr, type, count) _hfs_delete(ptr, type, count)

static inline __attribute__((const))
uint64_t hfs_blk_to_bytes(uint32_t blk, uint32_t blk_size)
{
	return (uint64_t)blk * blk_size; 		// Avoid the overflow
}

int32_t BTDeleteRecord(filefork_t    *filePtr,
					   BTreeIterator *iterator);

#define HFS_ALLOC_ROLL_BACK			0x800	//Reallocate blocks that were just deallocated
typedef uint32_t hfs_block_alloc_flags_t;

typedef struct hfs_alloc_extra_args hfs_alloc_extra_args_t;

static inline errno_t hfs_block_alloc(__unused hfsmount_t *hfsmp,
									  __unused HFSPlusExtentDescriptor *extent,
									  __unused hfs_block_alloc_flags_t flags,
									  __unused hfs_alloc_extra_args_t *extra_args)
{
	return ENOTSUP;
}

#define BlockDeallocate(m, b, c, f)		(int16_t)0
#define BTFlushPath(ff)					(int32_t)0

static inline int hfs_flushvolumeheader(__unused struct hfsmount *hfsmp, 
										__unused int waitfor, 
										__unused int altflush)
{
	return 0;
}

#define hfs_mark_inconsistent(m, r)		(void)0

static inline errno_t MacToVFSError(errno_t err)
{
	return err;
}

struct hfs_ext_iter;

uint32_t hfs_total_blocks(const HFSPlusExtentDescriptor *ext, int count);
errno_t hfs_ext_iter_next_group(struct hfs_ext_iter *iter);
errno_t hfs_ext_iter_update(struct hfs_ext_iter *iter,
							HFSPlusExtentDescriptor *extents,
							int count,
							HFSPlusExtentRecord cat_extents);
errno_t hfs_ext_iter_check_group(struct hfs_ext_iter *iter);

static inline uint32_t ff_allocblocks(filefork_t *ff)
{
	return ff->ff_blocks - ff->ff_unallocblocks;
}

#endif
