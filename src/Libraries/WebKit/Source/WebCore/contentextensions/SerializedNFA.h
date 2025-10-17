/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 16, 2023.
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

#if ENABLE(CONTENT_EXTENSIONS)

#include "ImmutableNFA.h"
#include <wtf/FileSystem.h>
#include <wtf/StdLibExtras.h>

namespace WebCore {
namespace ContentExtensions {

struct NFA;

class SerializedNFA {
public:
    static std::optional<SerializedNFA> serialize(NFA&&);
    SerializedNFA(SerializedNFA&&) = default;

    template<typename T>
    class Range {
    public:
        Range(std::span<const T> span)
            : m_span(span)
        { }
        const T* begin() const { return std::to_address(m_span.begin()); }
        const T* end() const { return std::to_address(m_span.end()); }
        size_t size() const { return m_span.size(); }
        const T* pointerAt(size_t i) const { return &m_span[i]; }
        T valueAt(size_t i) const
        {
            T value;
            memcpySpan(singleElementSpan(value), m_span.subspan(i, 1));
            return value;
        }
    private:
        std::span<const T> m_span;
    };

    const Range<ImmutableNFANode> nodes() const;
    const Range<ImmutableRange<char>> transitions() const;
    const Range<uint32_t> targets() const;
    const Range<uint32_t> epsilonTransitionsTargets() const;
    const Range<uint64_t> actions() const;

    uint32_t root() const
    {
        RELEASE_ASSERT(nodes().size());
        return 0;
    }

    struct ConstTargetIterator {
        const SerializedNFA& serializedNFA;
        uint32_t position;

        uint32_t operator*() const { return serializedNFA.targets().valueAt(position); }
        const uint32_t* operator->() const { return serializedNFA.targets().pointerAt(position); }

        bool operator==(const ConstTargetIterator& other) const
        {
            ASSERT(&serializedNFA == &other.serializedNFA);
            return position == other.position;
        }

        ConstTargetIterator& operator++()
        {
            ++position;
            return *this;
        }
    };

    struct IterableConstTargets {
        const SerializedNFA& serializedNFA;
        uint32_t targetStart;
        uint32_t targetEnd;

        ConstTargetIterator begin() const { return { serializedNFA, targetStart }; }
        ConstTargetIterator end() const { return { serializedNFA, targetEnd }; }
    };

    struct ConstRangeIterator {
        const SerializedNFA& serializedNFA;
        uint32_t position;

        bool operator==(const ConstRangeIterator& other) const
        {
            ASSERT(&serializedNFA == &other.serializedNFA);
            return position == other.position;
        }

        ConstRangeIterator& operator++()
        {
            ++position;
            return *this;
        }

        char first() const
        {
            return range()->first;
        }

        char last() const
        {
            return range()->last;
        }

        IterableConstTargets data() const
        {
            const ImmutableRange<char>* range = this->range();
            return { serializedNFA, range->targetStart, range->targetEnd };
        };

    private:
        const ImmutableRange<char>* range() const
        {
            return serializedNFA.transitions().pointerAt(position);
        }
    };

    struct IterableConstRange {
        const SerializedNFA& serializedNFA;
        uint32_t rangesStart;
        uint32_t rangesEnd;

        ConstRangeIterator begin() const { return { serializedNFA, rangesStart }; }
        ConstRangeIterator end() const { return { serializedNFA, rangesEnd }; }

#if CONTENT_EXTENSIONS_STATE_MACHINE_DEBUGGING
        void debugPrint() const
        {
            for (const auto& range : *this)
                WTFLogAlways("    %d-%d", range.first, range.last);
        }
#endif
    };

    IterableConstRange transitionsForNode(uint32_t nodeId) const
    {
        const auto* node = nodes().pointerAt(nodeId);
        return { *this, node->rangesStart, node->rangesEnd };
    }

private:
    struct Metadata {
        size_t nodesSize { 0 };
        size_t transitionsSize { 0 };
        size_t targetsSize { 0 };
        size_t epsilonTransitionsTargetsSize { 0 };
        size_t actionsSize { 0 };

        size_t nodesOffset { 0 };
        size_t transitionsOffset { 0 };
        size_t targetsOffset { 0 };
        size_t epsilonTransitionsTargetsOffset { 0 };
        size_t actionsOffset { 0 };
    };
    SerializedNFA(FileSystem::MappedFileData&&, Metadata&&);

    template<typename T>
    std::span<const T> spanAtOffsetInFile(size_t offset, size_t length) const;
    
    FileSystem::MappedFileData m_file;
    Metadata m_metadata;
};

} // namespace ContentExtensions
} // namespace WebCore

#endif // ENABLE(CONTENT_EXTENSIONS)
