/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 1, 2024.
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
#ifndef mach_o_RebaseOpcodes_h
#define mach_o_RebaseOpcodes_h

#include <stdint.h>
#include <stdio.h>

#include <span>

#include "Error.h"
#include "Header.h"
#include "Fixups.h"


namespace mach_o {

struct MappedSegment;

/*!
 * @class RebaseOpcodes
 *
 * @abstract
 *      Class to encapsulate accessing and building rebase opcodes
 */
class VIS_HIDDEN RebaseOpcodes
{
public:
                    // encapsulates rebase opcodes from a final linked image
                    RebaseOpcodes(const uint8_t* start, size_t size, bool is64);

    Error           valid(std::span<const MappedSegment> segments, bool allowTextFixups=false, bool onlyFixupsInWritableSegments=true) const;
    void            forEachRebaseLocation(std::span<const MappedSegment> segments, uint64_t prefLoadAdder, void (^callback)(const Fixup& fixup, bool& stop)) const;
    void            forEachRebaseLocation(void (^callback)(uint32_t segIndex, uint64_t segOffset, bool& stop)) const;
    void            printOpcodes(FILE* output, int indent=0) const;
    const uint8_t*  bytes(size_t& size) const;

private:
    struct SegRange { std::string_view segName; uint64_t vmSize; bool readable; bool writable; bool executable; };

    Error           forEachRebase(void (^handler)(const char* opcodeName, int type, bool segIndexSet,
                                                  uint8_t segmentIndex, uint64_t segmentOffset, bool& stop)) const;

protected:
    const uint8_t*       _opcodesStart;
    const uint8_t*       _opcodesEnd;
    const uint32_t       _pointerSize;
 };



} // namespace mach_o

#endif // mach_o_RebaseOpcodes_h


