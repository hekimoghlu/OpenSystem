/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 25, 2024.
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
#include "config.h"
#include "ObjectInitializationScope.h"

#include "HeapInlines.h"
#include "JSCJSValueInlines.h"
#include "JSCellInlines.h"
#include "JSObject.h"
#include "Scribble.h"

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

#if ASSERT_ENABLED

ObjectInitializationScope::ObjectInitializationScope(VM& vm)
    : m_vm(vm)
{
}

ObjectInitializationScope::~ObjectInitializationScope()
{
    m_vm.mutatorFence();
    if (!m_object)
        return;
    verifyPropertiesAreInitialized(m_object);
}

void ObjectInitializationScope::notifyAllocated(JSObject* object)
{
    ASSERT(!m_disallowGC);
    ASSERT(!m_disallowVMEntry);
    m_disallowGC.emplace();
    m_disallowVMEntry.emplace(m_vm);
    m_object = object;
}

void ObjectInitializationScope::notifyInitialized(JSObject* object)
{
    if (m_object) {
        m_disallowGC.reset();
        m_disallowVMEntry.reset();
        m_object = nullptr;
    }
    verifyPropertiesAreInitialized(object);
}

void ObjectInitializationScope::verifyPropertiesAreInitialized(JSObject* object)
{
    Butterfly* butterfly = object->butterfly();
    Structure* structure = object->structure();
    IndexingType indexingType = structure->indexingType();
    unsigned vectorLength = butterfly->vectorLength();
    if (UNLIKELY(hasUndecided(indexingType)) || !hasIndexedProperties(indexingType)) {
        // Nothing to verify.
    } else if (LIKELY(!hasAnyArrayStorage(indexingType))) {
        auto data = butterfly->contiguous().data();
        for (unsigned i = 0; i < vectorLength; ++i) {
            if (isScribbledValue(data[i].get())) {
                dataLogLn("Found scribbled value at i = ", i);
                ASSERT_NOT_REACHED();
            }
        }
    } else {
        ArrayStorage* storage = butterfly->arrayStorage();
        for (unsigned i = 0; i < vectorLength; ++i) {
            if (isScribbledValue(storage->m_vector[i].get())) {
                dataLogLn("Found scribbled value at i = ", i);
                ASSERT_NOT_REACHED();
            }
        }
    }

    auto isSafeEmptyValueForGCScanning = [] (JSValue value) {
#if USE(JSVALUE64)
        return !value;
#else
        return !value || !JSValue::encode(value);
#endif
    };

    for (int64_t i = 0; i < static_cast<int64_t>(structure->outOfLineCapacity()); i++) {
        // We rely on properties past the last offset be zero for concurrent GC.
        if (i + firstOutOfLineOffset > structure->maxOffset())
            ASSERT(isSafeEmptyValueForGCScanning(butterfly->propertyStorage()[-i - 1].get()));
        else if (isScribbledValue(butterfly->propertyStorage()[-i - 1].get())) {
            dataLogLn("Found scribbled property at i = ", -i - 1);
            ASSERT_NOT_REACHED();
        }
    }
}

#endif // ASSERT_ENABLED

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
