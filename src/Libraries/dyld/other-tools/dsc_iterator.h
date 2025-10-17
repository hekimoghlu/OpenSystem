/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 5, 2024.
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
#include <stdint.h>
#include <uuid/uuid.h>

struct dyld_shared_cache_dylib_info {
	uint32_t		version;		// current version 2
	// following fields all exist in version 1
	uint32_t		isAlias;		// this is alternate path (symlink)
	const void*		machHeader;		// of dylib in mapped cached file
	const char*		path;			// of dylib
	const uuid_t*	uuid;			// of dylib, or NULL is missing
	// following fields all exist in version 2
	uint64_t		inode;			// of dylib file or path hash
	uint64_t		modTime;		// of dylib file
};
typedef struct dyld_shared_cache_dylib_info dyld_shared_cache_dylib_info;

struct dyld_shared_cache_segment_info {
	uint64_t		version;		// initial version 1
	// following fields exist in version 1
	const char*		name;			// of segment
	uint64_t		fileOffset;		// of segment in cache file
	uint64_t		fileSize;		// of segment
	uint64_t		address;		// of segment when cache mapped with ASLR (sliding) off
    // following fields exist in version 2
	uint64_t		addressOffset;	// of segment from base of mapped cache
};
typedef struct dyld_shared_cache_segment_info dyld_shared_cache_segment_info;

#ifdef __cplusplus
extern "C" {
#endif
// Given a pointer and size of an in-memory copy of a dyld shared cache file,
// this routine will call the callback block once for each segment in each dylib 
// in the shared cache file.  
// Returns -1 if there was an error, otherwise 0.
extern int dyld_shared_cache_iterate(const void* shared_cache_file, uint32_t shared_cache_size,
									void (^callback)(const dyld_shared_cache_dylib_info* dylibInfo, const dyld_shared_cache_segment_info* segInfo));

#ifdef __cplusplus
}
#endif

