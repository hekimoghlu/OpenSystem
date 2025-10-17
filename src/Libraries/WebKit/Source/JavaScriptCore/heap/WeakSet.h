/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 27, 2024.
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

#include "CellContainer.h"
#include "WeakBlock.h"
#include <wtf/SentinelLinkedList.h>

namespace JSC {

class Heap;
class WeakImpl;

namespace Integrity {
class Analyzer;
}

class WeakSet : public BasicRawSentinelNode<WeakSet> {
    friend class LLIntOffsetsExtractor;
    friend class Integrity::Analyzer;

public:
    static WeakImpl* allocate(JSValue, WeakHandleOwner* = nullptr, void* context = nullptr);
    static void deallocate(WeakImpl*);

    WeakSet(VM&);
    ~WeakSet();
    void lastChanceToFinalize();
    
    JSC::Heap* heap() const;
    VM& vm() const;

    bool isEmpty() const;
    bool isTriviallyDestructible() const;

    void reap();
    void sweep();
    void shrink();
    void resetAllocator();

    static constexpr ptrdiff_t offsetOfVM() { return OBJECT_OFFSETOF(WeakSet, m_vm); }

    WeakBlock* head() { return m_blocks.head(); }

    template<typename Functor>
    void forEachBlock(const Functor& functor)
    {
        for (WeakBlock* block = m_blocks.head(); block; block = block->next())
            functor(*block);
    }

private:
    JS_EXPORT_PRIVATE WeakBlock::FreeCell* findAllocator(CellContainer);
    WeakBlock::FreeCell* tryFindAllocator();
    WeakBlock::FreeCell* addAllocator(CellContainer);
    void removeAllocator(WeakBlock*);

    WeakBlock::FreeCell* m_allocator { nullptr };
    WeakBlock* m_nextAllocator { nullptr };
    DoublyLinkedList<WeakBlock> m_blocks;
    // m_vm must be a pointer (instead of a reference) because the JSCLLIntOffsetsExtractor
    // cannot handle it being a reference.
    VM* const m_vm;
};

inline WeakSet::WeakSet(VM& vm)
    : m_vm(&vm)
{
}

inline VM& WeakSet::vm() const
{
    return *m_vm;
}

inline bool WeakSet::isEmpty() const
{
    for (WeakBlock* block = m_blocks.head(); block; block = block->next()) {
        if (!block->isEmpty())
            return false;
    }

    return true;
}

inline bool WeakSet::isTriviallyDestructible() const
{
    if (!m_blocks.isEmpty())
        return false;
    if (isOnList())
        return false;
    return true;
}

ALWAYS_INLINE void WeakSet::deallocate(WeakImpl* weakImpl)
{
    weakImpl->clear();
}

inline void WeakSet::lastChanceToFinalize()
{
    forEachBlock([](WeakBlock& block) {
        block.lastChanceToFinalize();
    });
}

inline void WeakSet::reap()
{
    forEachBlock([](WeakBlock& block) {
        block.reap();
    });
}

inline void WeakSet::resetAllocator()
{
    m_allocator = nullptr;
    m_nextAllocator = m_blocks.head();
}

} // namespace JSC
