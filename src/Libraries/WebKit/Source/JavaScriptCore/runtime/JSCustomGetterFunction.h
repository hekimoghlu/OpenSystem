/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 14, 2024.
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

#include "JSFunction.h"

namespace JSC {

class JSCustomGetterFunction final : public JSFunction {
public:
    typedef JSFunction Base;
    static constexpr unsigned StructureFlags = Base::StructureFlags;

    using CustomFunctionPointer = GetValueFunc;

    static constexpr DestructionMode needsDestruction = NeedsDestruction;
    static void destroy(JSCell*);

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.customGetterFunctionSpace<mode>();
    }

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    JS_EXPORT_PRIVATE static JSCustomGetterFunction* create(VM&, JSGlobalObject*, const PropertyName&, CustomFunctionPointer, std::optional<DOMAttributeAnnotation> = std::nullopt);

    DECLARE_EXPORT_INFO;

    const Identifier& propertyName() const { return m_propertyName; }
    CustomFunctionPointer getter() const { return m_getter; };
    CustomFunctionPointer customFunctionPointer() const { return m_getter; };
    std::optional<DOMAttributeAnnotation> domAttribute() const { return m_domAttribute; };
    const ClassInfo* slotBaseClassInfoIfExists() const
    {
        if (m_domAttribute)
            return m_domAttribute->classInfo;
        return nullptr;
    }

private:
    JSCustomGetterFunction(VM&, NativeExecutable*, JSGlobalObject*, Structure*, const PropertyName&, CustomFunctionPointer, std::optional<DOMAttributeAnnotation>);

    Identifier m_propertyName;
    CustomFunctionPointer m_getter;
    std::optional<DOMAttributeAnnotation> m_domAttribute;
};

} // namespace JSC
