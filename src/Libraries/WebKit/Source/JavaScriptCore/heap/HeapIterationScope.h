/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 4, 2022.
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
#include <wtf/Noncopyable.h>

namespace JSC {

class HeapIterationScope {
    WTF_MAKE_NONCOPYABLE(HeapIterationScope);
public:
    HeapIterationScope(Heap&);
    ~HeapIterationScope();

private:
    JSC::Heap& m_heap;
};

inline HeapIterationScope::HeapIterationScope(JSC::Heap& heap)
    : m_heap(heap)
{
    // FIXME: It would be nice to assert we're holding the API lock when iterating the heap so we know no other thread is mutating the heap
    // but adding `ASSERT_WITH_MESSAGE(heap.vm().currentThreadIsHoldingAPILock(), "Trying to iterate the JS heap without the API lock");`
    // causes spurious crashes since the only thing technically needed is just heap.hasAccess() but that doesn't verify this thread is
    // the one with access only that *some* thread has access.
    m_heap.willStartIterating();
}

inline HeapIterationScope::~HeapIterationScope()
{
    m_heap.didFinishIterating();
}

} // namespace JSC
