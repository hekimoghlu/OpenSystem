/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 17, 2024.
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
#include <stdio.h>
#include <string.h>

#include "SplitSeg.h"
#include "Misc.h"

// FIXME: We should get this from cctools
#define DYLD_CACHE_ADJ_V2_FORMAT 0x7F

namespace mach_o {

SplitSegInfo::SplitSegInfo(const uint8_t* start, size_t size)
: _infoStart(start), _infoEnd(start + size)
{
}

Error SplitSegInfo::valid() const
{
    return Error::none();
}

bool SplitSegInfo::hasMarker() const
{
    return (_infoStart == _infoEnd);
}

bool SplitSegInfo::isV1() const
{
    return !this->isV2();
}

bool SplitSegInfo::isV2() const
{
    return (*_infoStart == DYLD_CACHE_ADJ_V2_FORMAT);
}

Error SplitSegInfo::forEachReferenceV2(void (^callback)(const Entry& entry, bool& stop)) const
{
    const uint8_t* infoStart = this->_infoStart;
    const uint8_t* infoEnd = this->_infoEnd;

    if ( *infoStart++ != DYLD_CACHE_ADJ_V2_FORMAT ) {
        return Error("Not split seg v2");
    }

    // Whole         :== <count> FromToSection+
    // FromToSection :== <from-sect-index> <to-sect-index> <count> ToOffset+
    // ToOffset         :== <to-sect-offset-delta> <count> FromOffset+
    // FromOffset     :== <kind> <count> <from-sect-offset-delta>
    const uint8_t* p = infoStart;
    bool malformed = false;
    uint64_t sectionCount = read_uleb128(p, infoEnd, malformed);
    if ( malformed )
        return Error("malformed uleb128");
    for (uint64_t i=0; i < sectionCount; ++i) {
        uint64_t fromSectionIndex = read_uleb128(p, infoEnd, malformed);
        if ( malformed )
            return Error("malformed uleb128");
        uint64_t toSectionIndex = read_uleb128(p, infoEnd, malformed);
        if ( malformed )
            return Error("malformed uleb128");
        uint64_t toOffsetCount = read_uleb128(p, infoEnd, malformed);
        if ( malformed )
            return Error("malformed uleb128");
        uint64_t toSectionOffset = 0;
        for (uint64_t j=0; j < toOffsetCount; ++j) {
            uint64_t toSectionDelta = read_uleb128(p, infoEnd, malformed);
            if ( malformed )
                return Error("malformed uleb128");
            uint64_t fromOffsetCount = read_uleb128(p, infoEnd, malformed);
            if ( malformed )
                return Error("malformed uleb128");
            toSectionOffset += toSectionDelta;
            for (uint64_t k=0; k < fromOffsetCount; ++k) {
                uint64_t kind = read_uleb128(p, infoEnd, malformed);
                if ( malformed )
                    return Error("malformed uleb128");
                if ( kind > 13 ) {
                    return Error("bad kind (%llu) value\n", kind);
                }
                uint64_t fromSectDeltaCount = read_uleb128(p, infoEnd, malformed);
                if ( malformed )
                    return Error("malformed uleb128");
                uint64_t fromSectionOffset = 0;
                for (uint64_t l=0; l < fromSectDeltaCount; ++l) {
                    uint64_t delta = read_uleb128(p, infoEnd, malformed);
                    if ( malformed )
                        return Error("malformed uleb128");
                    fromSectionOffset += delta;
                    bool stop = false;
                    callback(Entry { (uint8_t)kind, (uint8_t)fromSectionIndex, (uint8_t)toSectionIndex, fromSectionOffset, toSectionOffset }, stop);
                    if ( stop )
                        return Error::none();
                }
            }
        }
    }

    return Error::none();
}

} // namepace mach_o
