/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 26, 2023.
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
#pragma once

#include "Opcode.h"
#include "ValueProfile.h"
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>

#include <wtf/SystemMalloc.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

class VM;

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(MetadataTable);
// using MetadataTableMalloc = SystemMalloc;
DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(UnlinkedMetadataTable);

class MetadataTable;

#if ENABLE(METADATA_STATISTICS)
struct MetadataStatistics {
    static size_t unlinkedMetadataCount;
    static size_t size32MetadataCount;
    static size_t totalMemory;
    static size_t perOpcodeCount[NUMBER_OF_BYTECODE_WITH_METADATA];
    static size_t numberOfCopiesFromLinking;
    static size_t linkingCopyMemory;

    static void reportMetadataStatistics();
};
#endif


class UnlinkedMetadataTable : public ThreadSafeRefCounted<UnlinkedMetadataTable> {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(UnlinkedMetadataTable);
    friend class LLIntOffsetsExtractor;
    friend class MetadataTable;
    friend class CachedMetadataTable;
#if ENABLE(METADATA_STATISTICS)
    friend struct MetadataStatistics;
#endif
public:
    static constexpr unsigned s_maxMetadataAlignment = 8;

    struct LinkingData {
        Ref<UnlinkedMetadataTable> unlinkedMetadata;
        std::atomic<unsigned> refCount;
    };

    ~UnlinkedMetadataTable();

    unsigned addEntry(OpcodeID);
    unsigned addValueProfile();

    size_t sizeInBytesForGC();

    void finalize();

    RefPtr<MetadataTable> link();

    static Ref<UnlinkedMetadataTable> create()
    {
        return adoptRef(*new UnlinkedMetadataTable);
    }

    template <typename Bytecode>
    unsigned numEntries();

    bool isFinalized() { return m_isFinalized; }
    bool hasMetadata() { return m_hasMetadata; }

    unsigned numValueProfiles() const { return m_numValueProfiles; }

    TriState didOptimize() const { return m_didOptimize; }
    void setDidOptimize(TriState didOptimize) { m_didOptimize = didOptimize; }

private:
    enum EmptyTag { Empty };

    UnlinkedMetadataTable();
    UnlinkedMetadataTable(bool is32Bit, unsigned numValueProfiles, unsigned lastOffset);
    UnlinkedMetadataTable(EmptyTag);

    static Ref<UnlinkedMetadataTable> create(bool is32Bit, unsigned numValueProfiles, unsigned lastOffset)
    {
        return adoptRef(*new UnlinkedMetadataTable(is32Bit, numValueProfiles, lastOffset));
    }

    static Ref<UnlinkedMetadataTable> empty()
    {
        return adoptRef(*new UnlinkedMetadataTable(Empty));
    }

    void unlink(MetadataTable&);

    size_t sizeInBytesForGC(MetadataTable&);

    unsigned totalSize() const
    {
        ASSERT(m_isFinalized);
        unsigned valueProfileSize = m_numValueProfiles * sizeof(ValueProfile);
        if (m_is32Bit)
            return valueProfileSize + offsetTable32()[s_offsetTableEntries - 1];
        return valueProfileSize + offsetTable16()[s_offsetTableEntries - 1];
    }

    unsigned offsetTableSize() const
    {
        ASSERT(m_isFinalized);
        if (m_is32Bit)
            return s_offset16TableSize + s_offset32TableSize;
        return s_offset16TableSize;
    }


    using Offset32 = uint32_t;
    using Offset16 = uint16_t;

    static constexpr unsigned s_offsetTableEntries = NUMBER_OF_BYTECODE_WITH_METADATA + 1; // one extra entry for the "end" offset;

    // Not to break alignment of 32bit offset table, we round up size with sizeof(Offset32).
    static constexpr unsigned s_offset16TableSize = roundUpToMultipleOf<sizeof(Offset32)>(s_offsetTableEntries * sizeof(Offset16));
    // Not to break alignment of the metadata calculated based on the alignment of s_offset16TableSize, s_offset32TableSize must be rounded by 8.
    // Then, s_offset16TableSize and s_offset16TableSize + s_offset32TableSize offer the same alignment characteristics for subsequent Metadata.
    static constexpr unsigned s_offset32TableSize = roundUpToMultipleOf<s_maxMetadataAlignment>(s_offsetTableEntries * sizeof(Offset32));

    void* buffer() const { return m_rawBuffer + m_numValueProfiles * sizeof(ValueProfile) + sizeof(LinkingData); }
    Offset32* preprocessBuffer() const { return std::bit_cast<Offset32*>(m_rawBuffer); }

    Offset16* offsetTable16() const
    {
        ASSERT(!m_is32Bit);
        return std::bit_cast<Offset16*>(m_rawBuffer + m_numValueProfiles * sizeof(ValueProfile) + sizeof(LinkingData));
    }
    Offset32* offsetTable32() const
    {
        ASSERT(m_is32Bit);
        return std::bit_cast<Offset32*>(m_rawBuffer + m_numValueProfiles * sizeof(ValueProfile) + sizeof(LinkingData) + s_offset16TableSize);
    }

    bool m_hasMetadata : 1;
    bool m_isFinalized : 1;
    bool m_isLinked : 1;
    bool m_is32Bit : 1;
    TriState m_didOptimize : 2 { TriState::Indeterminate };
    unsigned m_numValueProfiles { 0 };
    uint8_t* m_rawBuffer;
};

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
