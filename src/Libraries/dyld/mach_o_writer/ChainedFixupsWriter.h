/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 25, 2024.
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
#ifndef mach_o_writer_ChainedFixups_h
#define mach_o_writer_ChainedFixups_h

#include <stdint.h>
#include <stdio.h>
#include <mach-o/fixup-chains.h>

#include <span>
#include <vector>

// mach_o
#include "ChainedFixups.h"
#include "Error.h"
#include "Header.h"
#include "Fixups.h"

namespace mach_o {

using namespace mach_o;

/*!
 * @class ChainedFixups
 *
 * @abstract
 *      Class to encapsulate building chained fixups
 */
class VIS_HIDDEN ChainedFixupsWriter : public ChainedFixups
{
public:
    // Information we need to encode a single segment with chained fixups
    struct SegmentFixupsInfo {
        MappedSegment             mappedSegment;
        std::span<const Fixup>    fixups;
        uint32_t                  numPageExtras;
    };

    // used by unit tests to build chained fixups
    ChainedFixupsWriter(std::span<const Fixup::BindTarget> bindTargets,
                        std::span<const Fixup> fixups,
                        std::span<const MappedSegment> segments,
                        uint64_t preferredLoadAddress,
                        const PointerFormat& pf, uint32_t pageSize,
                        bool setDataChains,
                        bool startsInSection=false, bool useFileOffsets=false);

    // used by Layout to build chained fixups
    ChainedFixupsWriter(std::span<const Fixup::BindTarget> bindTargets,
                        std::span<const SegmentFixupsInfo> segments,
                        uint64_t preferredLoadAddress,
                        const PointerFormat& pf, uint32_t pageSize,
                        bool setDataChains,
                        bool startsInSection=false, bool useFileOffsets=false);


    static Error            importsFormat(std::span<const Fixup::BindTarget> bindTargets, uint16_t& importsFormat, size_t& stringPoolSize);

    static size_t           linkeditSize(std::span<const Fixup::BindTarget> bindTargets,
                                         std::span<const SegmentFixupsInfo> segments,
                                         const PointerFormat& pointerFormat, uint32_t pageSize);
    static size_t           startsSectionSize(std::span<const SegmentFixupsInfo> segments, const PointerFormat& pointerFormat);

    // Fills in the SegmentFixupsInfo::numPageExtras field for every segment with page extras
    static void             calculateSegmentPageExtras(std::span<SegmentFixupsInfo> segments,
                                                       const PointerFormat& pointerFormat,
                                                       uint32_t pageSize);

    Error           valid(uint64_t preferredLoadAddress, std::span<const MappedSegment> segments, bool startsInSection=false) const;

    const uint8_t*  bytes(size_t& size) const;

    void                                    buildLinkeditFixups(std::span<const Fixup::BindTarget> bindTargets,
                                                                std::span<const SegmentFixupsInfo> segments,
                                                                uint64_t preferredLoadAddress,
                                                                const PointerFormat& pf, uint32_t pageSize,
                                                                bool setDataChains);
    void                                    buildStartsSectionFixups(std::span<const SegmentFixupsInfo> segments,
                                                                     const PointerFormat& pf,
                                                                     bool useFileOffsets, uint64_t preferredLoadAddress);
    static uint32_t                         addSymbolString(CString symbolName, std::vector<char>& pool);

    Error                   _buildError;
    std::vector<uint8_t>    _bytes;
};


} // namespace mach_o

#endif // mach_o_writer_ChainedFixups_h


