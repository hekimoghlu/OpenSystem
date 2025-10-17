/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 12, 2023.
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
#ifndef mach_o_ExportsTrie_h
#define mach_o_ExportsTrie_h

#include <stdint.h>

#include <span>
#include <string_view>

#include "Array.h"

#include "Error.h"


namespace mach_o {

class Symbol;
struct GenericTrieWriterEntry;
struct GenericTrieNode;

/*!
 * @class GenericTrie
 *
 * @abstract
 *      Abstract base class for searching and building tries
 */
class VIS_HIDDEN GenericTrie
{
protected:
                    // construct from an already built trie
                    GenericTrie(const uint8_t* start, size_t size);

                    struct Entry { std::string_view name; std::span<const uint8_t> terminalPayload; };

    bool            hasEntry(const char* name, std::span<const uint8_t>& terminalPayload) const;
    void            forEachEntry(void (^callback)(const Entry& entry, bool& stop)) const;
    uint32_t        entryCount() const;

    void            dump() const;
    Error           recurseTrie(const uint8_t* p, dyld3::OverflowSafeArray<char>& cummulativeString,
                                int curStrOffset, bool& stop, void (^callback)(const char* name, std::span<const uint8_t> nodePayload, bool& stop)) const;

    const uint8_t*       _trieStart;
    const uint8_t*       _trieEnd;
};



/*!
 * @class ExportsTrie
 *
 * @abstract
 *      Class to encapsulate accessing and building export symbol tries
 */
class VIS_HIDDEN ExportsTrie : public GenericTrie
{
public:
                    // encapsulates exports trie in a final linked image
                    ExportsTrie(const uint8_t* start, size_t size) : GenericTrie(start, size) { }

    Error           valid(uint64_t maxVmOffset) const;
    bool            hasExportedSymbol(const char* symbolName, Symbol& symbol) const;
    void            forEachExportedSymbol(void (^callback)(const Symbol& symbol, bool& stop)) const;
    uint32_t        symbolCount() const;

private:
    Error           terminalPayloadToSymbol(const Entry& entry, Symbol& symInfo) const;
};


/*!
 * @class DylibsPathTrie
 *
 * @abstract
 *      Class to encapsulate accessing and building tries as in the dyld cache
 *      to map paths to dylib index.
 */
class VIS_HIDDEN DylibsPathTrie : public GenericTrie
{
public:
                    // encapsulates dylib path trie in the dyld cach
                    DylibsPathTrie(const uint8_t* start, size_t size) : GenericTrie(start, size) { }

                    struct DylibAndIndex { std::string_view path=""; uint32_t index=0; };
    
    bool            hasPath(const char* path, uint32_t& dylibIndex) const;
    void            forEachDylibPath(void (^callback)(const DylibAndIndex& info, bool& stop)) const;

private:
    bool            entryToIndex(std::span<const uint8_t> payload, uint32_t& index) const;
};


} // namespace mach_o

#endif // mach_o_ExportsTrie_h


