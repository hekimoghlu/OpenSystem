/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 13, 2023.
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
#ifndef mach_o_writer_ExportsTrie_h
#define mach_o_writer_ExportsTrie_h

#include <stdint.h>

#include <span>
#include <string_view>

#include <vector>
#include <list>

// mach_o
#include "Error.h"
#include "ExportsTrie.h"

// common
#include "ChunkBumpAllocator.h"

namespace mach_o
{
class Symbol;
}

namespace mach_o {

using namespace mach_o;

struct GenericTrieWriterEntry;
struct GenericTrieNode;

/*!
 * @class GenericTrieWriter
 *
 * @abstract
 *      Abstract base class for building tries
 */
class VIS_HIDDEN GenericTrieWriter : public GenericTrie
{
public:
                    // construct from an already built trie
                    GenericTrieWriter();

    const uint8_t*  bytes(size_t& size);
    size_t          size() { return _trieSize; }
    Error&          buildError() { return _buildError; }
    void            writeTrieBytes(std::span<uint8_t> bytes);

protected:
    void            buildNodes(std::span<const GenericTrieWriterEntry> entries);

    Error                _buildError;
    std::vector<uint8_t> _trieBytes;
    GenericTrieNode*     _rootNode=nullptr;
    size_t               _trieSize;

    ChunkBumpAllocatorZone _allocatorZone;
};



/*!
 * @class ExportsTrieWriter
 *
 * @abstract
 *      Class to encapsulate building export symbol tries
 */
class VIS_HIDDEN ExportsTrieWriter : public GenericTrieWriter
{
public:
                    // encapsulates exports trie in a final linked image
                    ExportsTrieWriter(std::span<const Symbol> exports, bool writeBytes=true, bool needsSort=true);

                    // build a trie from an existing trie, but filter out some entries
                    ExportsTrieWriter(const ExportsTrie& inputExportsTrie, bool (^remove)(const Symbol& sym));

    // From ExportsTrie
    bool            hasExportedSymbol(const char* symbolName, Symbol& symbol) const;
    void            forEachExportedSymbol(void (^callback)(const Symbol& symbol, bool& stop)) const;

    operator ExportsTrie() const;

    Error           valid(uint64_t maxVmOffset) const;
};


/*!
 * @class DylibsPathTrieWriter
 *
 * @abstract
 *      Class to encapsulate building tries as in the dyld cache
 *      to map paths to dylib index.
 */
class VIS_HIDDEN DylibsPathTrieWriter : public GenericTrieWriter
{
public:
                    // encapsulates dylib path trie in the dyld cache
                    DylibsPathTrieWriter(std::span<const mach_o::DylibsPathTrie::DylibAndIndex> dylibs, bool needsSort=true);

    // From DylibsPathTrie
    bool            hasPath(const char* path, uint32_t& dylibIndex) const;
    void            forEachDylibPath(void (^callback)(const mach_o::DylibsPathTrie::DylibAndIndex& info, bool& stop)) const;

    operator DylibsPathTrie() const;
};


} // namespace mach_o

#endif // mach_o_writer_ExportsTrie_h


