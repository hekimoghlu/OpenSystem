/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 24, 2024.
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

#include "CustomGetterSetter.h"
#include "DOMAnnotation.h"

namespace JSC {
namespace DOMJIT {

class GetterSetter;

}

class DOMAttributeGetterSetter final : public CustomGetterSetter {
public:
    using Base = CustomGetterSetter;

    template<typename CellType, SubspaceAccess>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return &vm.domAttributeGetterSetterSpace();
    }

    static DOMAttributeGetterSetter* create(VM& vm, CustomGetter customGetter, CustomSetter customSetter, DOMAttributeAnnotation domAttribute)
    {
        DOMAttributeGetterSetter* customGetterSetter = new (NotNull, allocateCell<DOMAttributeGetterSetter>(vm)) DOMAttributeGetterSetter(vm, customGetter, customSetter, domAttribute);
        customGetterSetter->finishCreation(vm);
        return customGetterSetter;
    }

    DOMAttributeAnnotation domAttribute() const { return m_domAttribute; }

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    DECLARE_EXPORT_INFO;

private:
    DOMAttributeGetterSetter(VM& vm, CustomGetter getter, CustomSetter setter, DOMAttributeAnnotation domAttribute)
        : Base(vm, vm.domAttributeGetterSetterStructure.get(), getter, setter)
        , m_domAttribute(domAttribute)
    {
    }

    DOMAttributeAnnotation m_domAttribute;
};

} // namespace JSC
