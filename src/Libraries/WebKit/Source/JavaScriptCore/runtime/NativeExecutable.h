/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 23, 2022.
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

#include "ExecutableBase.h"
#include "ImplementationVisibility.h"

namespace JSC {

class NativeExecutable final : public ExecutableBase {
    friend class JIT;
    friend class LLIntOffsetsExtractor;
public:
    typedef ExecutableBase Base;
    static constexpr unsigned StructureFlags = Base::StructureFlags | StructureIsImmortal;

    static NativeExecutable* create(VM&, Ref<JSC::JITCode>&& callThunk, TaggedNativeFunction, Ref<JSC::JITCode>&& constructThunk, TaggedNativeFunction constructor, ImplementationVisibility, const String& name);

    static void destroy(JSCell*);
    
    template<typename CellType, SubspaceAccess>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return &vm.nativeExecutableSpace();
    }

    CodeBlockHash hashFor(CodeSpecializationKind) const;

    TaggedNativeFunction function() const { return m_function; }
    TaggedNativeFunction constructor() const { return m_constructor; }
        
    TaggedNativeFunction nativeFunctionFor(CodeSpecializationKind kind)
    {
        if (kind == CodeForCall)
            return function();
        ASSERT(kind == CodeForConstruct);
        return constructor();
    }
        
    static constexpr ptrdiff_t offsetOfNativeFunctionFor(CodeSpecializationKind kind)
    {
        if (kind == CodeForCall)
            return OBJECT_OFFSETOF(NativeExecutable, m_function);
        ASSERT(kind == CodeForConstruct);
        return OBJECT_OFFSETOF(NativeExecutable, m_constructor);
    }

    DECLARE_VISIT_CHILDREN;
    static Structure* createStructure(VM&, JSGlobalObject*, JSValue proto);
        
    DECLARE_INFO;

    const String& name() const { return m_name; }

    const DOMJIT::Signature* signatureFor(CodeSpecializationKind) const;
    ImplementationVisibility implementationVisibility() const { return static_cast<ImplementationVisibility>(m_implementationVisibility); }
    Intrinsic intrinsic() const;

    JSString* toString(JSGlobalObject* globalObject)
    {
        if (!m_asString)
            return toStringSlow(globalObject);
        return m_asString.get();
    }

    JSString* asStringConcurrently() const { return m_asString.get(); }
    static constexpr ptrdiff_t offsetOfAsString() { return OBJECT_OFFSETOF(NativeExecutable, m_asString); }

private:
    NativeExecutable(VM&, TaggedNativeFunction, TaggedNativeFunction constructor, ImplementationVisibility);
    void finishCreation(VM&, Ref<JSC::JITCode>&& callThunk, Ref<JSC::JITCode>&& constructThunk, const String& name);

    JSString* toStringSlow(JSGlobalObject*);

    TaggedNativeFunction m_function;
    TaggedNativeFunction m_constructor;

    unsigned m_implementationVisibility : bitWidthOfImplementationVisibility;

    String m_name;
    WriteBarrier<JSString> m_asString;
};

} // namespace JSC
