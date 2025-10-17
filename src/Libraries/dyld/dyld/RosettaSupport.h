/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 10, 2023.
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
#ifndef RosettaSupport_h
#define RosettaSupport_h

//#include <unistd.h>
#include <stdint.h>

#include <mach-o/dyld_images.h>
#include <TargetConditionals.h>

#include "Defines.h"


#if SUPPORT_ROSETTA

#include <Rosetta/Dyld/Traps.h>

struct dyld_all_runtime_info {
    uint64_t                    image_count;
    const dyld_image_info*      images;
    uint64_t                    uuid_count;
    const dyld_uuid_info*       uuids;
    uint64_t                    aot_image_count;
    const dyld_aot_image_info*  aots;
    dyld_aot_shared_cache_info  aot_cache_info;
};

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

// Called once at launch to get AOT info about main executable
inline int aot_get_runtime_info(dyld_all_runtime_info*& info)
{
    return rosetta_dyld_get_runtime_info((const void**)&info);
}

// Called computing image size from disk to get info about translated mapping
inline int aot_get_extra_mapping_info(int fd, const char* path, uint64_t& extraAllocSize, char aotPath[], size_t aotPathSize)
{
    return rosetta_dyld_get_aot_size(fd, path, &extraAllocSize, aotPath, aotPathSize);
}

// Called when mmap()ing disk image, to add in translated mapping
inline int aot_map_extra(const char* path, const mach_header* mh, const void* mappingEnd, const mach_header*& aotMapping, uint64_t& aotMappingSize, uint8_t aotImageKey[32])
{
    return rosetta_dyld_map_aot(path, (uint64_t)mh, (uint64_t)mappingEnd, (uint64_t*)&aotMapping, (uint64_t*)&aotMappingSize, aotImageKey);
}

#pragma clang diagnostic pop



#endif /* SUPPORT_ROSETTA */

#endif /* RosettaSupport_h */
