/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 2, 2023.
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
#ifndef mach_o_writer_LinkerOptimizationHints_hpp
#define mach_o_writer_LinkerOptimizationHints_hpp

// common

// mach_o
#include "Error.h"
#include "LinkerOptimizationHints.h"

#include <stdint.h>
#include <stdio.h>

#include <span>
#include <vector>

namespace mach_o {
struct MappedSegment;
}

namespace mach_o {

using namespace mach_o;

/*!
 * @class LinkerOptimizationHintsWriter
 *
 * @abstract
 *      Class to encapsulate building linker optimization hints
 */
class VIS_HIDDEN LinkerOptimizationHintsWriter : public LinkerOptimizationHints
{
public:

    // used by unit tests to build LOHs
    struct Location
    {
        Location(Kind kind, std::vector<uint64_t> addrs);
        Location(Kind kind, std::span<uint64_t> addrs);

        Kind kind;
        std::vector<uint64_t> addrs;

        bool operator==(const Location& other) const
        {
            return kind == other.kind && addrs == other.addrs;
        }
    };

    LinkerOptimizationHintsWriter(std::span<const Location> sortedLocs, bool is64);

private:
    std::vector<uint8_t>        _bytes;

    void append_uleb128(uint64_t value);
};

} // namespace mach_o

#endif // mach_o_writer_LinkerOptimizationHints_hpp
