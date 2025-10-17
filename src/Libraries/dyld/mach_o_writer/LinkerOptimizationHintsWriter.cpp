/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 25, 2022.
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
// mach_o
#include "Fixups.h"
#include "Misc.h"

// mach_o_writer
#include "LinkerOptimizationHintsWriter.h"

namespace mach_o {

//
// MARK: --- LinkerOptimizationHints::Location methods ---
//

LinkerOptimizationHintsWriter::Location::Location(Kind kind, std::vector<uint64_t> addrs)
: kind(kind), addrs(addrs)
{
}

LinkerOptimizationHintsWriter::Location::Location(Kind kind, std::span<uint64_t> addrs)
: kind(kind), addrs(addrs.begin(), addrs.end())
{
}

void LinkerOptimizationHintsWriter::append_uleb128(uint64_t value)
{
    uint8_t byte;
    do {
        byte = value & 0x7F;
        value &= ~0x7F;
        if ( value != 0 )
            byte |= 0x80;
        _bytes.push_back(byte);
        value = value >> 7;
    } while( byte >= 0x80 );
}

LinkerOptimizationHintsWriter::LinkerOptimizationHintsWriter(std::span<const Location> sortedLocs, bool is64)
    : LinkerOptimizationHints()
{
    if ( sortedLocs.empty() )
        return;

    _bytes.reserve(256);
    for ( const Location& loc : sortedLocs ) {
        append_uleb128((uint64_t)loc.kind);
        append_uleb128((uint64_t)loc.addrs.size());
        for ( uint64_t addr : loc.addrs )
            append_uleb128(addr);
    }

    // align to pointer size
    uint32_t pointerSize = is64 ? 8 : 4;
    while ( (_bytes.size() % pointerSize) != 0 )
        _bytes.push_back(0);

    _buffer = _bytes;
}

} // namespace mach_o
