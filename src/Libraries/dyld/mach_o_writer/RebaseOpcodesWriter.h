/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 28, 2025.
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
#ifndef mach_o_writer_RebaseOpcodes_h
#define mach_o_writer_RebaseOpcodes_h

#include <stdint.h>
#include <stdio.h>

#include <span>
#include <vector>

#include "Error.h"
#include "Header.h"
#include "Fixups.h"

// mach_o
#include "RebaseOpcodes.h"

namespace mach_o {
struct MappedSegment;
}

namespace mach_o {

using namespace mach_o;

/*!
 * @class RebaseOpcodes
 *
 * @abstract
 *      Class to encapsulate building rebase opcodes
 */
class VIS_HIDDEN RebaseOpcodesWriter : public RebaseOpcodes
{
public:
                    // used by unit tests to build opcodes
                    struct Location { uint32_t segIndex; uint64_t segOffset; auto operator<=>(const Location&) const = default; };
                    RebaseOpcodesWriter(std::span<const Location> sortedLocs, bool is64);

private:
    std::vector<uint8_t> _opcodes;
    void                 append_uleb128(uint64_t value);
    void                 append_byte(uint8_t value);
 };



} // namespace mach_o

#endif // mach_o_writer_RebaseOpcodes_h


