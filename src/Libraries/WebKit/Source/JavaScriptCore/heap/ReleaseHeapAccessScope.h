/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 27, 2023.
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

// Almost all of the VM's code runs with "heap access". This means that the GC thread believes that
// the VM is messing with the heap in a way that would be unsafe for certain phases of the collector,
// like the weak reference fixpoint, stack scanning, and changing barrier modes. However, many long
// running operations inside the VM don't require heap access. For example, memcpying a typed array
// if a reference to it is on the stack is totally fine without heap access. Blocking on a futex is
// also fine without heap access. Releasing heap access for long-running code (in the case of futex
// wait, possibly infinitely long-running) ensures that the GC can finish a collection cycle while
// you are waiting.
class ReleaseHeapAccessScope {
public:
    ReleaseHeapAccessScope(JSC::Heap& heap)
        : m_heap(heap)
    {
        m_heap.releaseAccess();
    }
    
    ~ReleaseHeapAccessScope()
    {
        m_heap.acquireAccess();
    }

private:
    JSC::Heap& m_heap;
};

class ReleaseHeapAccessIfNeededScope {
public:
    ReleaseHeapAccessIfNeededScope(JSC::Heap& heap)
        : m_heap(heap)
    {
        hadHeapAccess = m_heap.hasAccess();
        if (hadHeapAccess)
            m_heap.releaseAccess();
    }

    ~ReleaseHeapAccessIfNeededScope()
    {
        if (hadHeapAccess)
            m_heap.acquireAccess();
    }

private:
    JSC::Heap& m_heap;
    bool hadHeapAccess { false };
};

} // namespace JSC

