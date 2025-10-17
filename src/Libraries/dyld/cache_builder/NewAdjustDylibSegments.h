/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 10, 2024.
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
#ifndef NewAdjustDylibSegments_hpp
#define NewAdjustDylibSegments_hpp

#include "SectionCoalescer.h"
#include "Types.h"

#include <stdint.h>
#include <unordered_map>
#include <string>
#include <vector>

namespace dyld3 {

struct MachOFile;

};

class Diagnostics;

// Represents a segment in a dylib/kext which is going to be moved in to a cache buffer
struct MovedSegment
{
    // Where is this segment in the source file
    InputDylibVMAddress inputVMAddress;
    // TODO: See if we need this.  In theory the inputVMSize might be greater that the cacheVMSize
    // if we remove sections from the segment, eg, deduplicating strings/GOTs/etc
    InputDylibVMSize inputVMSize;

    // Where is this segment in the cache
    uint8_t*        cacheLocation = nullptr;
    CacheVMAddress  cacheVMAddress;
    CacheVMSize     cacheVMSize;
    CacheFileOffset cacheFileOffset;
    CacheFileSize   cacheFileSize;

    // Each segment has its own ASLRTracker
    cache_builder::ASLR_Tracker* aslrTracker = nullptr;
};

// Represents a piece of LINKEDIT in a dylib/kext which is going to be moved in to a cache buffer
struct MovedLinkedit
{
    enum class Kind
    {
        symbolNList,
        symbolStrings,
        indirectSymbols,
        functionStarts,
        dataInCode,
        exportTrie,
        functionVariants,

        numKinds
    };

    Kind            kind;
    CacheFileOffset dataOffset;
    CacheFileSize   dataSize;
    uint8_t*        cacheLocation = nullptr;
};

struct NListInfo
{
    uint32_t localsStartIndex   = 0;
    uint32_t localsCount        = 0;
    uint32_t globalsStartIndex  = 0;
    uint32_t globalsCount       = 0;
    uint32_t undefsStartIndex   = 0;
    uint32_t undefsCount        = 0;
};

struct DylibSegmentsAdjustor
{
    DylibSegmentsAdjustor(std::vector<MovedSegment>&&                              movedSegments,
                          std::unordered_map<MovedLinkedit::Kind, MovedLinkedit>&& movedLinkedit,
                          NListInfo&                                               nlistInfo);

    // Map from input dylib VMAddr to cache dylib VMAddr
    CacheVMAddress adjustVMAddr(InputDylibVMAddress inputVMAddr) const;

    void adjustDylib(Diagnostics& diag, CacheVMAddress cacheBaseAddress,
                     dyld3::MachOFile* cacheMF, std::string_view dylibID,
                     const uint8_t* chainedFixupsStart, const uint8_t* chainedFixupsEnd,
                     const uint8_t* splitSegInfoStart, const uint8_t* splitSegInfoEnd,
                     const uint8_t* rebaseOpcodesStart, const uint8_t* rebaseOpcodesEnd,
                     const DylibSectionCoalescer* sectionCoalescer);

    const std::vector<MovedSegment>                              movedSegments;
    const std::unordered_map<MovedLinkedit::Kind, MovedLinkedit> movedLinkedit;
    const NListInfo                                              nlistInfo;
};

#endif /* NewAdjustDylibSegments_hpp */
