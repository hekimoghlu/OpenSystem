/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 25, 2025.
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
#include "JSWeakObjectRef.h"

#include "JSCInlines.h"

namespace JSC {

const ClassInfo JSWeakObjectRef::s_info = { "WeakRef"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSWeakObjectRef) };

void JSWeakObjectRef::finishCreation(VM& vm, JSCell* value)
{
    m_lastAccessVersion = vm.currentWeakRefVersion();
    m_value.set(vm, this, value);
    Base::finishCreation(vm);
}

template<typename Visitor>
void JSWeakObjectRef::visitChildrenImpl(JSCell* cell, Visitor& visitor)
{
    auto* thisObject = jsCast<JSWeakObjectRef*>(cell);
    ASSERT_GC_OBJECT_INHERITS(thisObject, info());
    Base::visitChildren(thisObject, visitor);
    // This doesn't need to be atomic because if we are out of date we will get write barriered and revisit ourselves.
    if (visitor.vm().currentWeakRefVersion() == thisObject->m_lastAccessVersion) {
        ASSERT(thisObject->m_value);
        visitor.append(thisObject->m_value);
    }
}

DEFINE_VISIT_CHILDREN(JSWeakObjectRef);

void JSWeakObjectRef::finalizeUnconditionally(VM& vm, CollectionScope)
{
    if (m_value && !vm.heap.isMarked(m_value.get()))
        m_value.clear();
}

}

