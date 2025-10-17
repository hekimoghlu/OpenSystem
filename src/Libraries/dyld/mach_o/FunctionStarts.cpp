/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 12, 2025.
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
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include "FunctionStarts.h"
#include "Misc.h"


namespace mach_o {


FunctionStarts::FunctionStarts(const uint8_t* start, size_t size)
  : _funcStartsBegin(start), _funcStartsEnd(start+size)
{
}

Error FunctionStarts::valid(uint64_t maxFuncOffset) const
{
    uint64_t runtimeOffset = 0;
    for (const uint8_t* p=_funcStartsBegin; p < _funcStartsEnd; ) {
        bool malformed;
        uint64_t value = read_uleb128(p, _funcStartsEnd, malformed);
        if ( malformed )
            return Error("malformed uleb128 in function-starts data");
        // a delta of zero marks end of functionStarts stream
        if ( value == 0 ) {
            while ( p < _funcStartsEnd ) {
                if ( *p++ != 0 )
                    return Error("padding at end of function-starts not all zeros");
            }
            return Error::none();
        }
        runtimeOffset += value;
        if ( runtimeOffset > maxFuncOffset )
            return Error("functions-starts has entry beyond end of TEXT");
    };
    return Error("functions-starts not zero terminated");
}

void FunctionStarts::forEachFunctionStart(uint64_t loadAddr, void (^callback)(uint64_t funcAddr)) const
{
    uint64_t runtimeOffset = 0;
    for (const uint8_t* p=_funcStartsBegin; p < _funcStartsEnd; ) {
        bool malformed;
        uint64_t value = read_uleb128(p, _funcStartsEnd, malformed);
        // a delta of zero marks end of functionStarts stream
        if ( malformed || (value == 0) )
            return;
        runtimeOffset += value;
        callback(loadAddr+runtimeOffset);
    };
}




} // namespace mach_o
