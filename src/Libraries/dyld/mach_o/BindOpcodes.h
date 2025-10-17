/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 3, 2025.
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
#ifndef mach_o_BindOpcodes_h
#define mach_o_BindOpcodes_h

#include <stdint.h>
#include <stdio.h>

#include <optional>
#include <span>

#include "Error.h"
#include "Header.h"
#include "Fixups.h"


namespace mach_o {

struct MappedSegment;

/*!
 * @class BindOpcodes
 *
 * @abstract
 *      Class to encapsulate accessing and building bind opcodes
 */
class VIS_HIDDEN BindOpcodes
{
public:
                    // encapsulates Bind opcodes from a final linked image
                    BindOpcodes(const uint8_t* start, size_t size, bool is64);

    struct BindTarget {
        const char* symbolName=nullptr;
        int         libOrdinal=0;
        bool        weakImport=false;
        bool        strongOverrideOfWeakDef = false;
        int64_t     addend=0;
    };

    struct LocAndTarget {
        bool operator==(const BindOpcodes::LocAndTarget& other) const;
        uint32_t           segIndex=0;
        uint64_t           segOffset=0;
        const BindTarget*  target=nullptr;
    };

    Error           valid(std::span<const MappedSegment> segments, uint32_t dylibCount, bool allowTextFixups=false, bool onlyFixupsInWritableSegments=true) const;
    void            forEachBindLocation(void (^callback)(const LocAndTarget&, bool& stop)) const;
    void            forEachBindTarget(void (^callback)(const Fixup::BindTarget& target, bool& stop), void (^strongHandler)(const char* symbolName)) const;
    uint32_t        forEachBindLocation(std::span<const MappedSegment> segments, uint32_t startBindOrdinal, void (^callback)(const Fixup& fixup, bool& stop)) const;
    void            printOpcodes(FILE* output, int indent=0) const;
    const uint8_t*  bytes(size_t& size) const;

protected:
    virtual bool                hasDoneBetweenBinds() const;
    virtual std::optional<int>  implicitLibraryOrdinal() const;
    struct SegRange { std::string_view segName; uint64_t vmSize; bool readable; bool writable; bool executable; };

    Error           forEachBind(void (^handler)(const char* opcodeName, int type, bool segIndexSet, uint8_t segmentIndex, uint64_t segmentOffset,
                                                bool libraryOrdinalSet, int libOrdinal, const char* symbolName, bool weakImport, int64_t addend,
                                                bool targetOrAddendChanged, bool& stop),
                                void (^strongHandler)(const char* symbolName)) const;

    const uint8_t*       _opcodesStart;
    const uint8_t*       _opcodesEnd;
    const uint32_t       _pointerSize;
};

class VIS_HIDDEN LazyBindOpcodes : public BindOpcodes
{
public:
                    // encapsulates Bind opcodes from a final linked image
                    LazyBindOpcodes(const uint8_t* start, size_t size, bool is64) : BindOpcodes(start, size, is64) { }

protected:
    bool            hasDoneBetweenBinds() const override final;

};

class VIS_HIDDEN WeakBindOpcodes : public BindOpcodes
{
public:
                    // encapsulates Bind opcodes from a final linked image
                    WeakBindOpcodes(const uint8_t* start, size_t size, bool is64) : BindOpcodes(start, size, is64) { }

protected:
    std::optional<int>  implicitLibraryOrdinal() const override final;
};


} // namespace mach_o

#endif // mach_o_BindOpcodes_h


