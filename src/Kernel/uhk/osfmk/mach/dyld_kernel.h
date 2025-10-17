/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 25, 2025.
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
#ifndef _MACH_DYLIB_INFO_H_
#define _MACH_DYLIB_INFO_H_

#include <mach/boolean.h>
#include <stdint.h>
#include <sys/_types/_fsid_t.h>
#include <sys/_types/_u_int32_t.h>
#include <sys/_types/_fsobj_id_t.h>
#include <sys/_types/_uuid_t.h>

/* These definitions must be kept in sync with the ones in
 * osfmk/mach/mach_types.defs.
 */

struct dyld_kernel_image_info {
	uuid_t uuid;
	fsobj_id_t fsobjid;
	fsid_t fsid;
	uint64_t load_addr;
};

struct dyld_kernel_process_info {
	struct dyld_kernel_image_info cache_image_info;
	uint64_t timestamp;         // mach_absolute_time of last time dyld change to image list
	uint32_t imageCount;        // number of images currently loaded into process
	uint32_t initialImageCount; // number of images statically loaded into process (before any dlopen() calls)
	uint8_t dyldState;          // one of dyld_process_state_* values
	boolean_t no_cache;         // process is running without a dyld cache
	boolean_t private_cache;    // process is using a private copy of its dyld cache
};

/* typedefs so our MIG is sane */

typedef struct dyld_kernel_image_info dyld_kernel_image_info_t;
typedef struct dyld_kernel_process_info dyld_kernel_process_info_t;
typedef dyld_kernel_image_info_t *dyld_kernel_image_info_array_t;

#endif /* _MACH_DYLIB_INFO_H_ */
