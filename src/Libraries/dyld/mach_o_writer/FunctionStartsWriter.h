/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 7, 2024.
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
#ifndef mach_o_writer_FunctionStarts_h
#define mach_o_writer_FunctionStarts_h

#include <span>
#include <stdint.h>

#include <vector>

// mach_o
#include "Error.h"
#include "FunctionStarts.h"

namespace mach_o {

using namespace mach_o;

/*!
 * @class FunctionStartsWriter
 *
 * @abstract
 *      Abstraction for building a list of function address in TEXT
 */
class VIS_HIDDEN FunctionStartsWriter : public FunctionStarts
{
public:
    // used build a function starts blob
    FunctionStartsWriter(uint64_t prefLoadAddr, std::span<const uint64_t> functionAddresses);

    std::span<const uint8_t>  bytes() const { return _bytes; }

private:
    void                  append_uleb128(uint64_t value);

    std::vector<uint8_t> _bytes;
};


} // namespace mach_o

#endif // mach_o_writer_FunctionStarts_h
