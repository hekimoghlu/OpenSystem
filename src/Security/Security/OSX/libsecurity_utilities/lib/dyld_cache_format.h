/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 15, 2024.
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
#ifndef __DYLD_CACHE_FORMAT__
#define __DYLD_CACHE_FORMAT__

#include <sys/types.h>
#include <stdint.h>
#include <mach/shared_region.h>


 struct dyld_cache_header
{
	char		magic[16];				// e.g. "dyld_v0     ppc"
	uint32_t	mappingOffset;			// file offset to first dyld_cache_mapping_info
	uint32_t	mappingCount;			// number of dyld_cache_mapping_info entries
	uint32_t	imagesOffset;			// file offset to first dyld_cache_image_info
	uint32_t	imagesCount;			// number of dyld_cache_image_info entries
	uint64_t	dyldBaseAddress;		// base address of dyld when cache was built
	uint64_t	codeSignatureOffset;	// file offset in of code signature blob
	uint64_t	codeSignatureSize;		// size of code signature blob (zero means to end of file)
};

struct dyld_cache_mapping_info {
	uint64_t	address;
	uint64_t	size;
	uint64_t	fileOffset;
	uint32_t	maxProt;
	uint32_t	initProt;
};

struct dyld_cache_image_info
{
	uint64_t	address;
	uint64_t	modTime;
	uint64_t	inode;
	uint32_t	pathFileOffset;
	uint32_t	pad;
};


#define DYLD_SHARED_CACHE_DIR			"/var/db/dyld/"
#define DYLD_SHARED_CACHE_BASE_NAME		"dyld_shared_cache_"



#endif // __DYLD_CACHE_FORMAT__


