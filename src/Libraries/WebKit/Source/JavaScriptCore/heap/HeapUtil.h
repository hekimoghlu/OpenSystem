/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 27, 2024.
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

namespace JSC {

// Are you tired of waiting for all of WebKit to build because you changed the implementation of a
// function in HeapInlines.h?  Does it bother you that you're waiting on rebuilding the JS DOM
// bindings even though your change is in a function called from only 2 .cpp files?  Then HeapUtil.h
// is for you!  Everything in this class should be a static method that takes a Heap& if needed.
// This is a friend of Heap, so you can access all of Heap's privates.
//
// This ends up being an issue because Heap exposes a lot of methods that ought to be inline for
// performance or that must be inline because they are templates.  This class ought to contain
// methods that are used for the implementation of the collector, or for unusual clients that need
// to reach deep into the collector for some reason.  Don't put things in here that would cause you
// to have to include it from more than a handful of places, since that would defeat the purpose.
// This class isn't here to look pretty.  It's to let us hack the GC more easily!

class HeapUtil {
public:
    static bool isPointerGCObjectJSCell(JSC::Heap& heap, TinyBloomFilter<uintptr_t> filter, JSCell* pointer)
    {
        // It could point to a large allocation.
        if (pointer->isPreciseAllocation()) {
            auto* set = heap.objectSpace().preciseAllocationSet();
            ASSERT(set);
            if (set->isEmpty())
                return false;
#if USE(JSVALUE32_64)
            // In 32bit systems a cell pointer can be 0xFFFFFFFF (an entries in the call frame), and this
            // value clashes with the deletedValue in a set<JSCell*>.
            if (!set->isValidValue(pointer))
                return false;
#endif
            return set->contains(pointer);
        }
    
        const UncheckedKeyHashSet<MarkedBlock*>& set = heap.objectSpace().blocks().set();
        
        MarkedBlock* candidate = MarkedBlock::blockFor(pointer);
        if (filter.ruleOut(std::bit_cast<uintptr_t>(candidate))) {
            ASSERT(!candidate || !set.contains(candidate));
            return false;
        }
        
        if (!MarkedBlock::isAtomAligned(pointer))
            return false;
        
        if (!set.contains(candidate))
            return false;
        
        if (candidate->handle().cellKind() != HeapCell::JSCell)
            return false;
        
        if (!candidate->handle().isLiveCell(pointer))
            return false;
        
        return true;
    }
    
    // This does not find the cell if the pointer is pointing at the middle of a JSCell.
    static bool isValueGCObject(
        JSC::Heap& heap, TinyBloomFilter<uintptr_t> filter, JSValue value)
    {
        ASSERT(heap.objectSpace().preciseAllocationSet());
        if (!value.isCell())
            return false;
        return isPointerGCObjectJSCell(heap, filter, value.asCell());
    }
};

} // namespace JSC

