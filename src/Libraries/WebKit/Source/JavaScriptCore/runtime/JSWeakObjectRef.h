/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 15, 2022.
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

#include "JSObject.h"

namespace JSC {

class JSWeakObjectRef final : public JSNonFinalObject {
public:
    using Base = JSNonFinalObject;

    DECLARE_EXPORT_INFO;

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    static JSWeakObjectRef* create(VM& vm, Structure* structure, JSCell* target)
    {
        JSWeakObjectRef* instance = new (NotNull, allocateCell<JSWeakObjectRef>(vm)) JSWeakObjectRef(vm, structure);
        instance->finishCreation(vm, target);
        return instance;
    }

    JSCell* deref(VM& vm)
    {
        if (m_value && vm.currentWeakRefVersion() != m_lastAccessVersion) {
            m_lastAccessVersion = vm.currentWeakRefVersion();
            // Perform a GC barrier here so we rescan this object and keep the object alive if we wouldn't otherwise.
            vm.writeBarrier(this);
        }

        return m_value.get();
    }

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.weakObjectRefSpace<mode>();
    }

    void finalizeUnconditionally(VM&, CollectionScope);
    DECLARE_VISIT_CHILDREN;

private:
    JSWeakObjectRef(VM& vm, Structure* structure)
        : Base(vm, structure)
    {
    }

    JS_EXPORT_PRIVATE void finishCreation(VM&, JSCell* value);

    uintptr_t m_lastAccessVersion;
    WriteBarrier<JSCell> m_value;
};

} // namespace JSC

