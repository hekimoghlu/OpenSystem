/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 10, 2024.
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

class JSIteratorHelper final : public JSInternalFieldObjectImpl<2> {
public:
    using Base = JSInternalFieldObjectImpl<2>;

    enum class Field : uint8_t {
        Generator = 0,
        UnderlyingIterator,
    };
    static_assert(numberOfInternalFields == 2);

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.iteratorHelperSpace<mode>();
    }

    static std::array<JSValue, numberOfInternalFields> initialValues()
    {
        return { {
            jsNull(),
            jsNull(),
        } };
    }

    const WriteBarrier<Unknown>& internalField(Field field) const { return Base::internalField(static_cast<uint32_t>(field)); }
    WriteBarrier<Unknown>& internalField(Field field) { return Base::internalField(static_cast<uint32_t>(field)); }

    static JSIteratorHelper* createWithInitialValues(VM&, Structure*);
    static JSIteratorHelper* create(VM&, Structure*, JSValue generator, JSValue underlyingIterator);
    static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    DECLARE_INFO;

    DECLARE_VISIT_CHILDREN;

private:
    JSIteratorHelper(VM&, Structure*);

    void finishCreation(VM&, JSValue generator, JSValue underlyingIterator);
};

STATIC_ASSERT_IS_TRIVIALLY_DESTRUCTIBLE(JSIteratorHelper);

JSC_DECLARE_HOST_FUNCTION(iteratorHelperPrivateFuncCreate);

} // namespace JSC
