/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 7, 2024.
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
#include <TargetConditionals.h>

#if !TARGET_OS_EXCLAVEKIT

#include <stdlib.h>
#include <stdio.h>
#include <Availability.h>


#include "dsc_iterator.h"
#define NO_ULEB
#include "DyldSharedCache.h"
#include "Header.h"

using mach_o::Header;


static void forEachDylibInCache(const void* shared_cache_file, void (^handler)(const dyld_cache_image_info* cachedDylibInfo, bool isAlias))
{
    const dyld_cache_header*        header      = (dyld_cache_header*)shared_cache_file;
    const dyld_cache_mapping_info*  mappings    = (dyld_cache_mapping_info*)((char*)shared_cache_file + header->mappingOffset);
    dyld_cache_image_info*          images      = (dyld_cache_image_info*)((char*)shared_cache_file + header->imagesOffsetOld);
    uint32_t                        imagesCount = header->imagesCountOld;
    if ( header->mappingOffset >= offsetof(dyld_cache_header, imagesCount) ) {
        images = (dyld_cache_image_info*)((char*)shared_cache_file + header->imagesOffset);
        imagesCount = header->imagesCount;
    }

    if ( mappings[0].fileOffset != 0 )
        return;
    uint64_t firstImageOffset = 0;
    uint64_t firstRegionAddress = mappings[0].address;
    for (uint32_t i=0; i < imagesCount; ++i) {
        uint64_t offset = images[i].address - firstRegionAddress;
        if ( firstImageOffset == 0 )
            firstImageOffset = offset;
        // skip over aliases.  This is no longer valid in newer caches
        //bool isAlias = false;//(dylibs[i].pathFileOffset < firstImageOffset);
        handler(&images[i], false);
    }
}


extern int dyld_shared_cache_iterate(const void* shared_cache_file, uint32_t shared_cache_size,
                                         void (^callback)(const dyld_shared_cache_dylib_info* dylibInfo, const dyld_shared_cache_segment_info* segInfo)) {
    const dyld_cache_header*       header             = (dyld_cache_header*)shared_cache_file;
    const dyld_cache_mapping_info* mappings           = (dyld_cache_mapping_info*)((char*)shared_cache_file + header->mappingOffset);
    const uint64_t                 unslideLoadAddress = mappings[0].address;

    __block int      result = 0;
    forEachDylibInCache(shared_cache_file, ^(const dyld_cache_image_info* cachedDylibInfo, bool isAlias) {
        uint64_t                    imageCacheOffset = cachedDylibInfo->address - unslideLoadAddress;
        const Header*               mh               = (Header*)((uint8_t*)shared_cache_file + imageCacheOffset);
        const char*                 dylibPath        = (char*)shared_cache_file + cachedDylibInfo->pathFileOffset;

        dyld_shared_cache_dylib_info dylibInfo;
        uuid_t                       uuid;
        dylibInfo.version    = 2;
        dylibInfo.machHeader = mh;
        dylibInfo.path       = dylibPath;
        dylibInfo.modTime    = cachedDylibInfo->modTime;
        dylibInfo.inode      = cachedDylibInfo->inode;
        dylibInfo.isAlias    = isAlias;
        mh->getUuid(uuid);
        dylibInfo.uuid       = &uuid;
        mh->forEachSegment(^(const Header::SegmentInfo& info, bool& stop) {
            if ( info.fileSize > info.vmsize ) {
                stop = true;
                return;
            }
            dyld_shared_cache_segment_info segInfo;
            segInfo.version       = 2;
            segInfo.name          = info.segmentName.data(); // FIXME: Might not be null terminated
            segInfo.fileOffset    = info.fileOffset;
            segInfo.fileSize      = info.vmsize;
            segInfo.address       = info.vmaddr;
            segInfo.addressOffset = info.vmaddr - unslideLoadAddress;
            callback(&dylibInfo, &segInfo);
        });
    });
    return result;
}

#endif // !TARGET_OS_EXCLAVEKIT
