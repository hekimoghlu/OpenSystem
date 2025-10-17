/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 12, 2022.
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
#ifndef mach_o_DataInCode_h
#define mach_o_DataInCode_h

#include <span>
#include <stdint.h>

#include "MachODefines.h"
#include "Error.h"

namespace mach_o {

/*!
 * @class DataInCode
 *
 * @abstract
 *      Class to encapsulate accessing and building data in code
 */
class VIS_HIDDEN DataInCode
{
public:
                        // construct from a chunk of LINKEDIT
                        DataInCode(const uint8_t* start, size_t size);

    struct Entry
    {
        // TODO: Implement this
    };

    Error   valid() const;

    static uint32_t     dataInCodeSize(bool is64);

protected:

    const uint8_t*       _dataInCodeStart;
    const uint8_t*       _dataInCodeEnd;
};


} // namespace mach_o

#endif // mach_o_CompactUnwind_h
