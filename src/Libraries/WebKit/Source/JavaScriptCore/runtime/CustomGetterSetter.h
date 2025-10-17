/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 2, 2024.
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

#include "JSCast.h"
#include "PropertySlot.h"
#include "PutPropertySlot.h"
#include "Structure.h"

namespace JSC {

class CustomGetterSetter : public JSCell {
public:
    using Base = JSCell;
    static constexpr unsigned StructureFlags = Base::StructureFlags | OverridesGetOwnPropertySlot | OverridesPut | StructureIsImmortal;

    using CustomGetter = GetValueFunc;
    using CustomSetter = PutValueFunc;

    template<typename CellType, SubspaceAccess>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return &vm.customGetterSetterSpace();
    }

    static CustomGetterSetter* create(VM& vm, CustomGetter customGetter, CustomSetter customSetter)
    {
        CustomGetterSetter* customGetterSetter = new (NotNull, allocateCell<CustomGetterSetter>(vm)) CustomGetterSetter(vm, vm.customGetterSetterStructure.get(), customGetter, customSetter);
        customGetterSetter->finishCreation(vm);
        return customGetterSetter;
    }

    CustomGetterSetter::CustomGetter getter() const { return m_getter; }
    CustomGetterSetter::CustomSetter setter() const { return m_setter; }

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);
        
    DECLARE_EXPORT_INFO;

    static bool getOwnPropertySlot(JSObject*, JSGlobalObject*, PropertyName, PropertySlot&) { RELEASE_ASSERT_NOT_REACHED(); return false; }
    static bool put(JSCell*, JSGlobalObject*, PropertyName, JSValue, PutPropertySlot&) { RELEASE_ASSERT_NOT_REACHED(); return false; }
    static bool putByIndex(JSCell*, JSGlobalObject*, unsigned, JSValue, bool) { RELEASE_ASSERT_NOT_REACHED(); return false; }
    static bool setPrototype(JSObject*, JSGlobalObject*, JSValue, bool) { RELEASE_ASSERT_NOT_REACHED(); return false; }
    static bool defineOwnProperty(JSObject*, JSGlobalObject*, PropertyName, const PropertyDescriptor&, bool) { RELEASE_ASSERT_NOT_REACHED(); return false; }
    static bool deleteProperty(JSCell*, JSGlobalObject*, PropertyName, DeletePropertySlot&) { RELEASE_ASSERT_NOT_REACHED(); return false; }

protected:
    CustomGetterSetter(VM& vm, Structure* structure, CustomGetter getter, CustomSetter setter)
        : JSCell(vm, structure)
        , m_getter(getter)
        , m_setter(setter)
    {
    }

private:
    CustomGetter m_getter;
    CustomSetter m_setter;
};

} // namespace JSC
