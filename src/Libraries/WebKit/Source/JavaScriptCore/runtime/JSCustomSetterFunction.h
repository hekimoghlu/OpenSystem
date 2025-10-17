/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 11, 2022.
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

class JSCustomSetterFunction final : public JSFunction {
public:
    typedef JSFunction Base;
    static constexpr unsigned StructureFlags = Base::StructureFlags;

    using CustomFunctionPointer = PutValueFunc;

    static constexpr DestructionMode needsDestruction = NeedsDestruction;
    static void destroy(JSCell*);

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.customSetterFunctionSpace<mode>();
    }

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    JS_EXPORT_PRIVATE static JSCustomSetterFunction* create(VM&, JSGlobalObject*, const PropertyName&, CustomFunctionPointer);

    DECLARE_EXPORT_INFO;

    const Identifier& propertyName() const { return m_propertyName; }
    CustomFunctionPointer setter() const { return m_setter; };
    CustomFunctionPointer customFunctionPointer() const { return m_setter; };
    const ClassInfo* slotBaseClassInfoIfExists() const { return nullptr; }

private:
    JSCustomSetterFunction(VM&, NativeExecutable*, JSGlobalObject*, Structure*, const PropertyName&, CustomFunctionPointer);

    Identifier m_propertyName;
    CustomFunctionPointer m_setter;
};

} // namespace JSC
