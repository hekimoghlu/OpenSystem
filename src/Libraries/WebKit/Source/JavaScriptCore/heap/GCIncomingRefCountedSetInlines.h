/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 1, 2024.
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

#include "GCIncomingRefCountedSet.h"
#include "VM.h"

namespace JSC {

template<typename T>
GCIncomingRefCountedSet<T>::GCIncomingRefCountedSet()
    : m_bytes(0)
{
}

template<typename T>
void GCIncomingRefCountedSet<T>::lastChanceToFinalize()
{
    for (size_t i = m_vector.size(); i--;)
        m_vector[i]->filterIncomingReferences([] (JSCell*) { return false; });
}

template<typename T>
bool GCIncomingRefCountedSet<T>::addReference(JSCell* cell, T* object)
{
    if (!object->addIncomingReference(cell)) {
        ASSERT(object->isDeferred());
        ASSERT(object->numberOfIncomingReferences());
        return false;
    }
    m_vector.append(object);
    m_bytes += object->gcSizeEstimateInBytes();
    ASSERT(object->isDeferred());
    ASSERT(object->numberOfIncomingReferences());
    return true;
}

template<typename T>
void GCIncomingRefCountedSet<T>::sweep(VM& vm, CollectionScope collectionScope)
{
    size_t preciseBytes = 0;
    m_vector.removeAllMatching([&](T* object) {
        size_t size = object->gcSizeEstimateInBytes();
        ASSERT(object->isDeferred());
        ASSERT(object->numberOfIncomingReferences());
        if (!object->filterIncomingReferences([&] (JSCell* cell) { return vm.heap.isMarked(cell); })) {
            preciseBytes += size;
            return false;
        }
        return true;
    });
    // Update m_bytes to the precise value when Full-GC happens since Eden-GC only expects that Eden region is collected.
    if (collectionScope == CollectionScope::Full)
        m_bytes = preciseBytes;
}

} // namespace JSC
