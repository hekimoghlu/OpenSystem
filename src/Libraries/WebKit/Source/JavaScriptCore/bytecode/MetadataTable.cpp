/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 26, 2025.
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
#include "config.h"
#include "MetadataTable.h"

#include "JSCJSValueInlines.h"
#include "OpcodeInlines.h"
#include "UnlinkedMetadataTableInlines.h"

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

MetadataTable::MetadataTable(UnlinkedMetadataTable& unlinkedMetadata)
{
    new (&linkingData()) UnlinkedMetadataTable::LinkingData {
        unlinkedMetadata,
        1,
    };
}

struct DeallocTable {
    template<typename Op>
    static void withOpcodeType(MetadataTable* table)
    {
        if constexpr (static_cast<unsigned>(Op::opcodeID) < NUMBER_OF_BYTECODE_WITH_METADATA) {
            table->forEach<Op>([](auto& entry) {
                entry.~Metadata();
            });
        }
    }
};

MetadataTable::~MetadataTable()
{
    for (unsigned i = 0; i < NUMBER_OF_BYTECODE_WITH_METADATA; i++)
        getOpcodeType<DeallocTable>(static_cast<OpcodeID>(i), this);
    linkingData().~LinkingData();
}

void MetadataTable::destroy(MetadataTable* table)
{
    // FIXME: This check should really not be necessary, see https://webkit.org/b/272787
    if (table->isDestroyed()) {
        ASSERT_NOT_REACHED();
        return;
    }

    RefPtr<UnlinkedMetadataTable> unlinkedMetadata = WTFMove(table->linkingData().unlinkedMetadata);
    ASSERT(table->isDestroyed());

    table->~MetadataTable();

    // Since UnlinkedMetadata::unlink frees the underlying memory of MetadataTable.
    // We need to destroy LinkingData before calling it.
    unlinkedMetadata->unlink(*table);
}

size_t MetadataTable::sizeInBytesForGC()
{
    return unlinkedMetadata()->sizeInBytesForGC(*this);
}

void MetadataTable::validate() const
{
    auto getOffset = [&](unsigned i) {
        unsigned offset = offsetTable16()[i];
        if (offset)
            return offset;
        return offsetTable32()[i];
    };
    UNUSED_PARAM(getOffset);
    ASSERT(getOffset(0) >= (is32Bit() ? UnlinkedMetadataTable::s_offset16TableSize + UnlinkedMetadataTable::s_offset32TableSize : UnlinkedMetadataTable::s_offset16TableSize));
    for (unsigned i = 1; i < UnlinkedMetadataTable::s_offsetTableEntries; ++i)
        ASSERT(getOffset(i-1) <= getOffset(i));
}

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
