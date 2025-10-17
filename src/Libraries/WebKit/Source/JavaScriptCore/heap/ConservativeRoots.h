/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 1, 2023.
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

#include "Heap.h"

namespace JSC {

class CodeBlockSet;
class HeapCell;
class JITStubRoutineSet;

class ConservativeRoots {
public:
    ConservativeRoots(Heap&);
    ~ConservativeRoots();

    void add(void* begin, void* end);
    void add(void* begin, void* end, JITStubRoutineSet&, CodeBlockSet&);
    
    size_t size() const;
    HeapCell** roots() const;

private:
    static constexpr size_t inlineCapacity = 2048;
    
    template<bool lookForWasmCallees, typename MarkHook>
    void genericAddPointer(char*, HeapVersion markingVersion, HeapVersion newlyAllocatedVersion, TinyBloomFilter<uintptr_t> jsGCFilter, TinyBloomFilter<uintptr_t> boxedWasmCalleeFilter, MarkHook&);

    template<typename MarkHook>
    void genericAddSpan(void* begin, void* end, MarkHook&);
    
    void grow();

    // We can't just use the copy of Heap::m_wasmCalleesPendingDestruction since new callees could be registered while
    // we're actively scanning the stack. A bad race would be:
    // 1) Start scanning the stack passing a frame with Wasm::Callee foo.
    // 2) tier up finishes for foo and is added to Heap::m_wasmCalleesPendingDestruction
    // 3) foo isn't added to m_wasmCalleesDiscovered
    // 4) foo gets derefed and destroyed.
    UncheckedKeyHashSet<const Wasm::Callee*> m_wasmCalleesPendingDestructionCopy;
    UncheckedKeyHashSet<const Wasm::Callee*> m_wasmCalleesDiscovered;
    TinyBloomFilter<uintptr_t> m_boxedWasmCalleeFilter;
    HeapCell** m_roots;
    size_t m_size;
    size_t m_capacity;
    JSC::Heap& m_heap;
    HeapCell* m_inlineRoots[inlineCapacity];
};

inline size_t ConservativeRoots::size() const
{
    return m_size;
}

inline HeapCell** ConservativeRoots::roots() const
{
    return m_roots;
}

} // namespace JSC
