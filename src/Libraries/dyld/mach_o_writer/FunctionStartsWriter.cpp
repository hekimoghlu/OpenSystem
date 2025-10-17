/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 1, 2022.
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
#include <assert.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

// mach_o
#include "Misc.h"

// mach_o_writer
#include "FunctionStartsWriter.h"


namespace mach_o {

FunctionStartsWriter::FunctionStartsWriter(uint64_t prefLoadAddr, std::span<const uint64_t> functionAddresses) : FunctionStarts(nullptr, 0)
{
    uint64_t lastAddr = prefLoadAddr;
    for (uint64_t addr : functionAddresses) {
        assert(addr >= lastAddr && "function addresses not sorted");
        // <rdar://problem/10422823> filter out zero-length atoms, so LC_FUNCTION_STARTS address can't spill into next section
        if ( addr == lastAddr)
            continue;
        // FIXME: for 32-bit arm need to check thumbness
        uint64_t delta = addr - lastAddr;
        append_uleb128(delta);
        lastAddr = addr;
    }
    // terminate delta encoded list
    _bytes.push_back(0);
    // 8-byte align
    while ( (_bytes.size() % 8) != 0 )
        _bytes.push_back(0);

    // set up pointers to data can be parsed
    _funcStartsBegin = _bytes.data();
    _funcStartsEnd   = _bytes.data()+_bytes.size();
}

void FunctionStartsWriter::append_uleb128(uint64_t value)
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




} // namespace mach_o
