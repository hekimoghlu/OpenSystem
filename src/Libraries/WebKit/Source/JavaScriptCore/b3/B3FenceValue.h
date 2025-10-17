/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 1, 2021.
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

#if ENABLE(B3_JIT)

#include "B3HeapRange.h"
#include "B3Value.h"

namespace JSC { namespace B3 {

class JS_EXPORT_PRIVATE FenceValue final : public Value {
public:
    static bool accepts(Kind kind) { return kind == Fence; }
    
    ~FenceValue() final;
    
    // The read/write heaps are reflected in the effects() of this value. The compiler may change
    // the lowering of a Fence based on the heaps. For example, if a fence does not write anything
    // then it is understood to be a store-store fence. On x86, this may lead us to not emit any
    // code, while on ARM we may emit a cheaper fence (dmb ishst instead of dmb ish). We will do
    // the same optimization for load-load fences, which are expressed as a Fence that writes but
    // does not read.
    //
    // This abstraction allows us to cover all of the fences on x86 and all of the standalone fences
    // on ARM. X86 really just has one fence: mfence. This fence should be used to protect stores
    // from being sunk below loads. WTF calls it the storeLoadFence. A classic example is the Steele
    // barrier:
    //
    //     o.f = v  =>  o.f = v
    //                  if (color(o) == black)
    //                      log(o)
    //
    // We are trying to ensure that if the store to o.f occurs after the collector has started
    // visiting o, then we will log o. Under sequential consistency, this would work. The collector
    // would set color(o) to black just before it started visiting. But x86's illusion of sequential
    // consistency is broken in exactly just this store->load ordering case. The store to o.f may
    // get buffered, and it may occur some time after we have loaded and checked color(o). As well,
    // the collector's store to set color(o) to black may get buffered and it may occur some time
    // after the collector has finished visiting o. Therefore, we need mfences. In B3 we model this
    // as a Fence that reads and writes some heaps. Setting writes to the empty set will cause B3 to
    // not emit any barrier on x86.
    //
    // On ARM there are many more fences. The Fence instruction is meant to model just two of them:
    // dmb ish and dmb ishst. You can emit a dmb ishst by using a Fence with an empty write heap.
    // Otherwise, you will get a dmb ish.
    // FIXME: Add fenced memory accesses. https://bugs.webkit.org/show_bug.cgi?id=162349
    // FIXME: Add a Depend operation. https://bugs.webkit.org/show_bug.cgi?id=162350
    HeapRange read { HeapRange::top() };
    HeapRange write { HeapRange::top() };

    B3_SPECIALIZE_VALUE_FOR_NO_CHILDREN

private:
    friend class Procedure;
    friend class Value;
    
    static Opcode opcodeFromConstructor(Origin, HeapRange = HeapRange(), HeapRange = HeapRange()) { return Fence; }
    FenceValue(Origin origin, HeapRange read, HeapRange write);
    FenceValue(Origin origin);
};

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)

