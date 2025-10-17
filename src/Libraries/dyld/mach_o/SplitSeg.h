/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 16, 2023.
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
#ifndef mach_o_SplitSeg_h
#define mach_o_SplitSeg_h

#include <span>
#include <stdint.h>

#include "MachODefines.h"
#include "Error.h"

#define DYLD_CACHE_ADJ_V2_FORMAT 0x7F

#define DYLD_CACHE_ADJ_V2_POINTER_32                0x01
#define DYLD_CACHE_ADJ_V2_POINTER_64                0x02
#define DYLD_CACHE_ADJ_V2_DELTA_32                  0x03
#define DYLD_CACHE_ADJ_V2_DELTA_64                  0x04
#define DYLD_CACHE_ADJ_V2_ARM64_ADRP                0x05
#define DYLD_CACHE_ADJ_V2_ARM64_OFF12               0x06
#define DYLD_CACHE_ADJ_V2_ARM64_BR26                0x07
#define DYLD_CACHE_ADJ_V2_ARM_MOVW_MOVT             0x08
#define DYLD_CACHE_ADJ_V2_ARM_BR24                  0x09
#define DYLD_CACHE_ADJ_V2_THUMB_MOVW_MOVT           0x0A
#define DYLD_CACHE_ADJ_V2_THUMB_BR22                0x0B
#define DYLD_CACHE_ADJ_V2_IMAGE_OFF_32              0x0C
#define DYLD_CACHE_ADJ_V2_THREADED_POINTER_64       0x0D

namespace mach_o {

/*!
 * @class SplitSegInfo
 *
 * @abstract
 *      Class to encapsulate accessing and building split seg info
 */
class VIS_HIDDEN SplitSegInfo
{
public:
                        // construct from a chunk of LINKEDIT
                        SplitSegInfo(const uint8_t* start, size_t size);

    struct Entry   { uint8_t kind; uint8_t fromSectionIndex; uint8_t toSectionIndex; uint64_t fromSectionOffset; uint64_t toSectionOffset; };

    Error   valid() const;
    bool    hasMarker() const;
    bool    isV1() const;
    bool    isV2() const;

    Error   forEachReferenceV2(void (^callback)(const Entry& entry, bool& stop)) const;

    static uint32_t     splitSegInfoSize(bool is64);

protected:

    const uint8_t*       _infoStart;
    const uint8_t*       _infoEnd;
};


} // namespace mach_o

#endif // mach_o_CompactUnwind_h
