/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 24, 2022.
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
#ifndef _UAPI_LINUX_MMAN_H
#define _UAPI_LINUX_MMAN_H
#include <asm/mman.h>
#include <asm-generic/hugetlb_encode.h>
#include <linux/types.h>
#define MREMAP_MAYMOVE 1
#define MREMAP_FIXED 2
#define MREMAP_DONTUNMAP 4
#define OVERCOMMIT_GUESS 0
#define OVERCOMMIT_ALWAYS 1
#define OVERCOMMIT_NEVER 2
#define MAP_SHARED 0x01
#define MAP_PRIVATE 0x02
#define MAP_SHARED_VALIDATE 0x03
#define MAP_DROPPABLE 0x08
#define MAP_HUGE_SHIFT HUGETLB_FLAG_ENCODE_SHIFT
#define MAP_HUGE_MASK HUGETLB_FLAG_ENCODE_MASK
#define MAP_HUGE_16KB HUGETLB_FLAG_ENCODE_16KB
#define MAP_HUGE_64KB HUGETLB_FLAG_ENCODE_64KB
#define MAP_HUGE_512KB HUGETLB_FLAG_ENCODE_512KB
#define MAP_HUGE_1MB HUGETLB_FLAG_ENCODE_1MB
#define MAP_HUGE_2MB HUGETLB_FLAG_ENCODE_2MB
#define MAP_HUGE_8MB HUGETLB_FLAG_ENCODE_8MB
#define MAP_HUGE_16MB HUGETLB_FLAG_ENCODE_16MB
#define MAP_HUGE_32MB HUGETLB_FLAG_ENCODE_32MB
#define MAP_HUGE_256MB HUGETLB_FLAG_ENCODE_256MB
#define MAP_HUGE_512MB HUGETLB_FLAG_ENCODE_512MB
#define MAP_HUGE_1GB HUGETLB_FLAG_ENCODE_1GB
#define MAP_HUGE_2GB HUGETLB_FLAG_ENCODE_2GB
#define MAP_HUGE_16GB HUGETLB_FLAG_ENCODE_16GB
struct cachestat_range {
  __u64 off;
  __u64 len;
};
struct cachestat {
  __u64 nr_cache;
  __u64 nr_dirty;
  __u64 nr_writeback;
  __u64 nr_evicted;
  __u64 nr_recently_evicted;
};
#endif
