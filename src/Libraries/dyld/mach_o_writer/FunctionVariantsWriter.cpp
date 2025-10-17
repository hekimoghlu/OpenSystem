/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 29, 2023.
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
#include "Defines.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>


#include "FunctionVariantsWriter.h"

namespace mach_o {


//
// MARK: --- FunctionVariantsRuntimeTableWriter methods ---
//

FunctionVariantsRuntimeTableWriter* FunctionVariantsRuntimeTableWriter::make(Kind kind, size_t variantsCount)
{
    size_t size = offsetof(FunctionVariantsRuntimeTable, entries[variantsCount]);
    FunctionVariantsRuntimeTableWriter* p = (FunctionVariantsRuntimeTableWriter*)::calloc(size, 1);
    p->kind  = kind;
    p->count = (uint32_t)variantsCount;
    return p;
}

Error FunctionVariantsRuntimeTableWriter::setEntry(size_t index, uint32_t impl, bool implIsTableIndex, std::span<const uint8_t> flagIndexes)
{
    if ( index >= this->count )
        return Error("index=%lu too large (max=%d)", index, this->count);
    if ( flagIndexes.size() > 4 )
        return Error("flagIndexes too large %lu (max 4)", flagIndexes.size());
    this->entries[index] = { impl, implIsTableIndex };
    memcpy(this->entries[index].flagBitNums, flagIndexes.data(), flagIndexes.size());
    return Error::none();
}


//
// MARK: --- FunctionVariantsWriter methods ---
//


FunctionVariantsWriter::FunctionVariantsWriter(std::span<const FunctionVariantsRuntimeTable*> entries)
{
    // compute size of linkedit blob to hold all FunctionVariantsRuntimeTable
    const size_t firstOffset = sizeof(OnDiskFormat) + entries.size()*sizeof(uint32_t);
    size_t       size        = firstOffset;
    for ( const FunctionVariantsRuntimeTable* fvrt : entries )
        size += fvrt->size();
    size = (size+7) & (-8);  // LINKEDIT content must be pointer size aligned

    // allocate byte vector to hold whole blob
    _builtBytes.resize(size);
    _bytes = _builtBytes;

    // fill in blob header and all entries
    OnDiskFormat* p             = header();
    uint32_t      currentOffset = (uint32_t)firstOffset;
    for ( const FunctionVariantsRuntimeTable* fvrt : entries ) {
        p->tableOffsets[p->tableCount] = currentOffset;
        p->tableCount++;
        size_t entrySize = fvrt->size();
        assert(currentOffset+entrySize <= size);
        memcpy(&_builtBytes[currentOffset], fvrt, entrySize);
        currentOffset += entrySize;
    }
}


//
// MARK: --- FunctionVariantFixupsWriter methods ---
//

FunctionVariantFixupsWriter::FunctionVariantFixupsWriter(std::span<const InternalFixup> entries)
{
    _builtBytes.resize(entries.size() * sizeof(InternalFixup));
    memcpy(_builtBytes.data(), entries.data(), entries.size() * sizeof(InternalFixup));
    _fixups = std::span<const InternalFixup>((InternalFixup*)_builtBytes.data(), entries.size());
}


} // namespace mach_o
