/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 27, 2022.
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
#ifndef macho_LinkerOptimizationHints_hpp
#define macho_LinkerOptimizationHints_hpp

// common
#include "MachODefines.h"

// mach_o
#include "Error.h"

#include <stdint.h>
#include <stdio.h>

#include <span>

namespace mach_o {

struct MappedSegment;

/*!
 * @class LinkerOptimizationHints
 *
 * @abstract
 *      Class to encapsulate accessing and building linker optimization hints
 */
class VIS_HIDDEN LinkerOptimizationHints
{
public:
    enum class Kind
    {
        unknown = 0,
        // 1 - 8 are actual LOHs we don't use any more

    };

    // construct from LC_LINKER_OPTIMIZATION_HINT range in .o file
    LinkerOptimizationHints(std::span<const uint8_t> buffer);

    Error           valid(std::span<const MappedSegment> segments, uint64_t loadAddress) const;
    Error           forEachLOH(void (^callback)(Kind kind, std::span<uint64_t> addrs, bool& stop)) const;
    void            printLOHs(FILE* output, int indent=0) const;
    const uint8_t*  bytes(size_t& size) const;

protected:
    // for use by LinkerOptimizationHintsWriter
    LinkerOptimizationHints() = default;

    std::span<const uint8_t>    _buffer;
};

} // namespace mach_o

#endif // macho_LinkerOptimizationHints_hpp
