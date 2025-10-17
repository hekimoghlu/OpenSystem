/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 15, 2024.
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

#include "CallFrame.h"
#include "JSCJSValue.h"
#include "JSCast.h"
#include "Structure.h"

namespace JSC {

class JSAPIValueWrapper final : public JSCell {
    friend JSValue jsAPIValueWrapper(JSGlobalObject*, JSValue);
public:
    using Base = JSCell;
    static constexpr unsigned StructureFlags = Base::StructureFlags | StructureIsImmortal;

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.apiValueWrapperSpace<mode>();
    }

    JSValue value() const { return m_value.get(); }

    static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    DECLARE_EXPORT_INFO;

    static JSAPIValueWrapper* create(VM& vm, JSValue value)
    {
        JSAPIValueWrapper* wrapper = new (NotNull, allocateCell<JSAPIValueWrapper>(vm)) JSAPIValueWrapper(vm, value);
        wrapper->finishCreation(vm);
        ASSERT(!value.isCell());
        return wrapper;
    }

private:
    JSAPIValueWrapper(VM& vm, JSValue value)
        : JSCell(vm, vm.apiWrapperStructure.get())
        , m_value(value, WriteBarrierEarlyInit)
    {
    }

    DECLARE_DEFAULT_FINISH_CREATION;

    WriteBarrier<Unknown> m_value;
};

} // namespace JSC
