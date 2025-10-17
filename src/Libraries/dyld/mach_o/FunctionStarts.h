/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 7, 2022.
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
#ifndef mach_o_FunctionStarts_h
#define mach_o_FunctionStarts_h

#include <span>
#include <stdint.h>

#include "MachODefines.h"
#include "Error.h"

namespace mach_o {

/*!
 * @class FunctionStarts
 *
 * @abstract
 *      Abstraction for a list of function address in TEXT
 */
class VIS_HIDDEN FunctionStarts
{
public:
                        // construct from a mach-o linkedit blob
                        FunctionStarts(const uint8_t* start, size_t size);

    Error               valid(uint64_t maxFuncOffset) const;
    void                forEachFunctionStart(uint64_t loadAddr, void (^callback)(uint64_t funcAddr)) const;

protected:
    const uint8_t*       _funcStartsBegin;
    const uint8_t*       _funcStartsEnd;
};


} // namespace mach_o

#endif // mach_o_FunctionStarts_h
