/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 7, 2022.
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
#ifndef mach_o_writer_SymbolTable_h
#define mach_o_writer_SymbolTable_h

#include <stdint.h>
#include <mach-o/nlist.h>

#include <span>
#include <vector>

// mach_o
#include "Error.h"
#include "MemoryBuffer.h"
#include "NListSymbolTable.h"
#include "Symbol.h"

#ifndef N_LIB
#define N_LIB    0x68
#endif

namespace mach_o
{
struct DebugNoteFileInfo;
}

namespace mach_o {

using namespace mach_o;

/*!
 * @class NListSymbolTable
 *
 * @abstract
 *      Class to encapsulate building an nlist symbol table in mach-o
 */
class VIS_HIDDEN NListSymbolTableWriter : public NListSymbolTable
{
public:
    struct DebugBuilderNoteItem
    {
        uint64_t        addr=0;
        uint64_t        size=0;

        union {
            // when using convenience NList constructor this has to point to
            // the note's name, otherwise the pointer won't be used, so callers
            // can use the `userData` field to store their own context.
            // ld's layout uses this to store atoms and implement efficient reuse
            // of the string pool strings.
            void*       userData=nullptr;
            const char* name;
        };

        uint8_t         type=0;
        uint8_t         sectNum=0;
        uint32_t        stringPoolOffset=0;
    };

    struct DebugBuilderNote
    {
        const DebugNoteFileInfo*            fileInfo;
        std::vector<DebugBuilderNoteItem>   items;
        uint32_t                            srcDirPoolOffset=0;
        uint32_t                            srcNamePoolOffset=0;
        uint32_t                            originLibPathPoolOffset=0;
        uint32_t                            objPathPoolOffset=0;
    };

    struct NListLayout
    {
        std::span<const Symbol>             globals;
        std::span<const uint32_t>           globalsStrx;
        std::span<const uint32_t>           reexportStrx;
        std::span<const Symbol>             undefs;
        std::span<const uint32_t>           undefsStrx;
        std::span<const Symbol>             locals;
        std::span<const uint32_t>           localsStrx;
        std::span<const DebugBuilderNote>   debugNotes;
        uint32_t                            debugNotesNListCount=0;
    };

                    // Convenience NList constructors used in unit tests
                    // \{
                    NListSymbolTableWriter(std::span<const Symbol> symbols, uint64_t prefLoadAddr, bool is64, std::span<DebugBuilderNote> debugNotes={}, bool zeroModTimes=false,
                                     bool objectFile=false);
                    NListSymbolTableWriter(std::span<const Symbol> globals, std::span<const Symbol> undefs, std::span<const Symbol> locals, std::span<DebugBuilderNote> debugNotes, uint64_t prefLoadAddr, bool is64, bool zeroModTimes);
                    // \}

                    // NList constructor with a precomputed layout and nlist buffer
                    NListSymbolTableWriter(NListLayout layout, std::span<uint8_t> nlistBuffer, uint64_t prefLoadAddr, bool is64, bool zeroModTimes);

    static uint32_t countDebugNoteNLists(std::span<const DebugBuilderNote> debugNotes);
    struct nlist    nlistFromSymbol(const Symbol&, uint32_t strx, uint32_t reexportStrx);
    struct nlist_64 nlist64FromSymbol(const Symbol&, uint32_t strx, uint32_t reexportStrx);

private:

    struct SymbolPartition
    {
        std::vector<Symbol>     locals;
        std::vector<Symbol>     globals;
        std::vector<Symbol>     undefs;

        SymbolPartition(std::span<const Symbol> symbol, bool objectFile);
    };

    struct NListBuffer
    {

        WritableMemoryBuffer storage;
        std::span<uint8_t> buffer;

        NListBuffer(std::span<uint8_t> buffer): buffer(buffer) {}
        NListBuffer(std::span<nlist_64> buffer): buffer((uint8_t*)buffer.data(), buffer.size_bytes()) {}
        NListBuffer(std::span<struct nlist> buffer): buffer((uint8_t*)buffer.data(), buffer.size_bytes()) {}

        NListBuffer(size_t bufferSize)
        {
            storage = WritableMemoryBuffer::allocate(bufferSize);
            buffer = storage;
        }

        NListBuffer() {}


        void add(nlist_64 nlist)
        {
            assert(buffer.size() >= (sizeof(nlist_64)));
            *(nlist_64*)buffer.data() = nlist;
            buffer = buffer.subspan(sizeof(nlist_64));
        }

        void add(struct nlist nlist)
        {
            assert(buffer.size() >= (sizeof(struct nlist)));
            *(struct nlist*)buffer.data() = nlist;
            buffer = buffer.subspan(sizeof(struct nlist));
        }
    };

                    NListSymbolTableWriter(const SymbolPartition& partition, std::span<DebugBuilderNote> debugNotes,
                            uint64_t prefLoadAddr, bool is64, bool zeroModTimes);

                    NListSymbolTableWriter(NListLayout layout, NListBuffer nlist, std::vector<char> stringPool,
                            uint64_t prefLoadAddr, bool is64, bool zeroModTimes);

    template <typename T>
    void            addStabsFromDebugNotes(std::span<const DebugBuilderNote> debugNotes, bool zeroModTimes, NListBuffer& nlists);

    NListBuffer             _nlistBuffer;
    std::vector<char>       _stringPoolBuffer;
};

} // namespace mach_o

#endif // mach_o_writer_SymbolTable_h


