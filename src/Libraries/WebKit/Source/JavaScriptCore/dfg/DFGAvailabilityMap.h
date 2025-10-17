/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 17, 2025.
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

#if ENABLE(DFG_JIT)

#include "DFGAvailability.h"
#include "DFGPromotedHeapLocation.h"

namespace JSC { namespace DFG {

struct AvailabilityMap {
    void pruneHeap();
    void pruneByLiveness(Graph&, CodeOrigin);
    void clear();
    
    void dump(PrintStream& out) const;
    
    friend bool operator==(const AvailabilityMap&, const AvailabilityMap&) = default;
    
    void merge(const AvailabilityMap& other);
    
    template<typename Functor>
    void forEachAvailability(const Functor& functor) const
    {
        for (unsigned i = m_locals.size(); i--;)
            functor(m_locals[i]);
        for (auto pair : m_heap)
            functor(pair.value);
    }
    
    template<typename HasFunctor, typename AddFunctor>
    void closeOverNodes(const HasFunctor& has, const AddFunctor& add) const
    {
        bool changed;
        do {
            changed = false;
            for (auto pair : m_heap) {
                if (pair.value.hasNode() && has(pair.key.base()))
                    changed |= add(pair.value.node());
            }
        } while (changed);
    }
    
    template<typename HasFunctor, typename AddFunctor>
    void closeStartingWithLocal(Operand op, const HasFunctor& has, const AddFunctor& add) const
    {
        Availability availability = m_locals.operand(op);
        if (!availability.hasNode())
            return;
        
        if (!add(availability.node()))
            return;
        
        closeOverNodes(has, add);
    }
    
    Operands<Availability> m_locals;
    UncheckedKeyHashMap<PromotedHeapLocation, Availability> m_heap;
};

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)
