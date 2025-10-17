/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 30, 2024.
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

#include "JSInternalFieldObjectImpl.h"

namespace JSC {

const static uint8_t JSAsyncFromSyncIteratorNumberOfInternalFields = 2;

class JSAsyncFromSyncIterator final : public JSInternalFieldObjectImpl<JSAsyncFromSyncIteratorNumberOfInternalFields> {
public:
    using Base = JSInternalFieldObjectImpl<JSAsyncFromSyncIteratorNumberOfInternalFields>;

    DECLARE_EXPORT_INFO;

    enum class Field : uint8_t {
        SyncIterator = 0,
        NextMethod,
    };
    static_assert(numberOfInternalFields == JSAsyncFromSyncIteratorNumberOfInternalFields);

    static std::array<JSValue, numberOfInternalFields> initialValues()
    {
        return { {
            jsNull(),
            jsNull(),
        } };
    }

    const WriteBarrier<Unknown>& internalField(Field field) const { return Base::internalField(static_cast<uint32_t>(field)); }
    WriteBarrier<Unknown>& internalField(Field field) { return Base::internalField(static_cast<uint32_t>(field)); }

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.asyncFromSyncIteratorSpace<mode>();
    }

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    static JSAsyncFromSyncIterator* createWithInitialValues(VM&, Structure*);
    static JSAsyncFromSyncIterator* create(VM&, Structure*, JSValue syncIterator, JSValue nextMethod);

    void setSyncIterator(VM& vm, JSValue syncIterator) { internalField(Field::SyncIterator).set(vm, this, syncIterator); }
    void setNextMethod(VM& vm, JSValue nextMethod) { internalField(Field::NextMethod).set(vm, this, nextMethod); }

private:
    JSAsyncFromSyncIterator(VM& vm, Structure* structure)
        : Base(vm, structure)
    {
    }

    void finishCreation(VM&, JSValue syncIterator, JSValue nextMethod);
    DECLARE_VISIT_CHILDREN;

};

STATIC_ASSERT_IS_TRIVIALLY_DESTRUCTIBLE(JSAsyncFromSyncIterator);

JSC_DECLARE_HOST_FUNCTION(asyncFromSyncIteratorPrivateFuncCreate);

} // namespace JSC
