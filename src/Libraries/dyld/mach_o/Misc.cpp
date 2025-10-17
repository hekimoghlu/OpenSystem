/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 1, 2025.
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
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <TargetConditionals.h>

#include <AvailabilityMacros.h>
#include <mach-o/dyld_introspection.h>
#include <mach-o/dyld_priv.h>

#include "Misc.h"
#include "SupportedArchs.h"
#include "Universal.h"
#include "Archive.h"
#include "Header.h"

namespace mach_o {

////////////////////////////  ULEB128 helpers  ////////////////////////////////////////

uint64_t read_uleb128(std::span<const uint8_t>& buffer, bool& malformed)
{
    uint64_t result = 0;
    int         bit = 0;
    malformed = false;
    while ( true ) {
        if ( buffer.empty() ) {
            malformed = true;
            break;
        }
        uint8_t elt = buffer.front();
        uint64_t slice = elt & 0x7f;

        if ( bit > 63 ) {
            malformed = true;
            break;
        }
        else {
            result |= (slice << bit);
            bit += 7;
        }

        // Keep iterating if the high bit is set, and advance to next element
        buffer = buffer.subspan(1);
        if ( (elt & 0x80) == 0 )
            break;
    }
    return result;
}

uint64_t read_uleb128(const uint8_t*& p, const uint8_t* end, bool& malformed)
{
    // Use the std::span one
    std::span<const uint8_t> buffer(p, end);
    uint64_t result = read_uleb128(buffer, malformed);

    // Adjust 'p' to the new start of the buffer as read_uleb128() would have advanced it
    if ( buffer.empty() )
        p = end;
    else
        p = &buffer.front();

    return result;
}

int64_t  read_sleb128(const uint8_t*& p, const uint8_t* end, bool& malformed)
{
    int64_t  result = 0;
    int      bit = 0;
    uint8_t  byte = 0;
    malformed = false;
    do {
        if ( p == end ) {
            malformed = true;
            break;
        }
        byte = *p++;
        result |= (((int64_t)(byte & 0x7f)) << bit);
        bit += 7;
    } while (byte & 0x80);
    // sign extend negative numbers
    if ( ((byte & 0x40) != 0) && (bit < 64) )
        result |= (~0ULL) << bit;
    return result;
}

uint32_t uleb128_size(uint64_t value)
{
    uint32_t result = 0;
    do {
        value = value >> 7;
        ++result;
    } while ( value != 0 );
    return result;
}

Error forEachHeader(std::span<const uint8_t> buffer, std::string_view path,
                    void (^callback)(const Header* sliceHeader, size_t sliceLength, bool& stop)) {
    if ( const mach_o::Universal* universal = mach_o::Universal::isUniversal(buffer) ) {
        if ( mach_o::Error err = universal->valid(buffer.size()) )
            return Error("error in file '%s': %s", path.data(), err.message());
        universal->forEachSlice(^(mach_o::Universal::Slice slice, bool &stop) {
            if ( const mach_o::Header* mh = mach_o::Header::isMachO(slice.buffer) ) {
                callback(mh, slice.buffer.size(), stop);
            }
        });
    } else if ( const mach_o::Header* mh = mach_o::Header::isMachO(buffer) ) {
        bool stop = false;
        callback(mh, buffer.size(), stop);
    }

    return Error::none();
}


} // namespace mach_o





