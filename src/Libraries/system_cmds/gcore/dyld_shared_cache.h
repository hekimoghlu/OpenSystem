/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 16, 2025.
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
#include <inttypes.h>
#include <stdbool.h>
#include <uuid/uuid.h>
#include <sys/types.h>

#ifndef _DYLD_SHARED_CACHE_H
#define _DYLD_SHARED_CACHE_H

#define IPHONE_DYLD_SHARED_CACHE_DIR_GCORE "/System/Cryptexes/OS/System/Library/Caches/com.apple.dyld/"
/*
 * Guilty knowledge of dyld shared cache internals, used to verify shared cache
 */
struct copied_dyld_cache_header {
    char		magic[16];		// e.g. "dyld_v0  i386"
    uint32_t	mappingOffset;		// file offset to first dyld_cache_mapping_info
    uint32_t	mappingCount;		// number of dyld_cache_mapping_info entries
    uint32_t	imagesOffset;		// file offset to first dyld_cache_image_info
    uint32_t	imagesCount;		// number of dyld_cache_image_info entries
    uint64_t	dyldBaseAddress;	// base address of dyld when cache was built
    uint64_t	codeSignatureOffset;	// file offset of code signature blob
    uint64_t	codeSignatureSize;	// size of code signature blob (zero means to end of file)
    uint64_t	slideInfoOffset;	// file offset of kernel slid info
    uint64_t	slideInfoSize;		// size of kernel slid info
    uint64_t	localSymbolsOffset;	// file offset of where local symbols are stored
    uint64_t	localSymbolsSize;	// size of local symbols information
    uint8_t		uuid[16];		// unique value for each shared cache file
    uint64_t	cacheType;		// 1 for development, 0 for optimized
};

extern bool get_uuid_from_shared_cache_mapping(const void *, size_t, uuid_t);
extern char *shared_cache_filename(const uuid_t);

#endif /* _DYLD_SHARED_CACHE_H */
