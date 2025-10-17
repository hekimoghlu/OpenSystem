/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 23, 2024.
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

#include "ErrorType.h"
#include "JSString.h"
#include "PrivateName.h"
#include <wtf/Expected.h>

namespace JSC {

class Symbol final : public JSCell {
public:
    typedef JSCell Base;
    static constexpr unsigned StructureFlags = Base::StructureFlags | StructureIsImmortal | OverridesPut;

    DECLARE_EXPORT_INFO;

    static constexpr DestructionMode needsDestruction = NeedsDestruction;

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.symbolSpace<mode>();
    }

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    static Symbol* create(VM&);
    static Symbol* createWithDescription(VM&, const String&);
    JS_EXPORT_PRIVATE static Symbol* create(VM&, SymbolImpl& uid);

    SymbolImpl& uid() const { return m_privateName.uid(); }
    PrivateName privateName() const { return m_privateName; }
    String descriptiveString() const;
    String description() const;
    Expected<String, ErrorTypeWithExtension> tryGetDescriptiveString() const;

    JSValue toPrimitive(JSGlobalObject*, PreferredPrimitiveType) const;
    JSObject* toObject(JSGlobalObject*) const;
    double toNumber(JSGlobalObject*) const;

    static constexpr ptrdiff_t offsetOfSymbolImpl()
    {
        // PrivateName is just a Ref<SymbolImpl> which can just be used as a SymbolImpl*.
        return OBJECT_OFFSETOF(Symbol, m_privateName);
    }

private:
    static void destroy(JSCell*);

    Symbol(VM&);
    Symbol(VM&, const String&);
    Symbol(VM&, SymbolImpl& uid);

    void finishCreation(VM&);

    PrivateName m_privateName;
};

Symbol* asSymbol(JSValue);

inline Symbol* asSymbol(JSValue value)
{
    ASSERT(value.asCell()->isSymbol());
    return jsCast<Symbol*>(value.asCell());
}

} // namespace JSC
