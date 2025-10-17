/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 29, 2023.
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
#include "config.h"
#include "NativeExecutable.h"

#include "Debugger.h"
#include "ExecutableBaseInlines.h"
#include "JSCInlines.h"
#include "VMInlines.h"

namespace JSC {

const ClassInfo NativeExecutable::s_info = { "NativeExecutable"_s, &ExecutableBase::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(NativeExecutable) };

NativeExecutable* NativeExecutable::create(VM& vm, Ref<JSC::JITCode>&& callThunk, TaggedNativeFunction function, Ref<JSC::JITCode>&& constructThunk, TaggedNativeFunction constructor, ImplementationVisibility implementationVisibility, const String& name)
{
    NativeExecutable* executable;
    executable = new (NotNull, allocateCell<NativeExecutable>(vm)) NativeExecutable(vm, function, constructor, implementationVisibility);
    executable->finishCreation(vm, WTFMove(callThunk), WTFMove(constructThunk), name);

    vm.forEachDebugger([&] (Debugger& debugger) {
        debugger.didCreateNativeExecutable(*executable);
    });

    return executable;
}

void NativeExecutable::destroy(JSCell* cell)
{
    static_cast<NativeExecutable*>(cell)->NativeExecutable::~NativeExecutable();
}

Structure* NativeExecutable::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue proto)
{
    return Structure::create(vm, globalObject, proto, TypeInfo(NativeExecutableType, StructureFlags), info());
}

void NativeExecutable::finishCreation(VM& vm, Ref<JSC::JITCode>&& callThunk, Ref<JSC::JITCode>&& constructThunk, const String& name)
{
    Base::finishCreation(vm);
    m_jitCodeForCall = WTFMove(callThunk);
    m_jitCodeForConstruct = WTFMove(constructThunk);
    m_jitCodeForCallWithArityCheck = m_jitCodeForCall->addressForCall(MustCheckArity);
    m_jitCodeForConstructWithArityCheck = m_jitCodeForConstruct->addressForCall(MustCheckArity);
    m_name = name;

    assertIsTaggedWith<JSEntryPtrTag>(m_jitCodeForCall->addressForCall(ArityCheckNotRequired).taggedPtr());
    assertIsTaggedWith<JSEntryPtrTag>(m_jitCodeForConstruct->addressForCall(ArityCheckNotRequired).taggedPtr());
    assertIsTaggedWith<JSEntryPtrTag>(m_jitCodeForCallWithArityCheck.taggedPtr());
    assertIsTaggedWith<JSEntryPtrTag>(m_jitCodeForConstructWithArityCheck.taggedPtr());
}

NativeExecutable::NativeExecutable(VM& vm, TaggedNativeFunction function, TaggedNativeFunction constructor, ImplementationVisibility implementationVisibility)
    : ExecutableBase(vm, vm.nativeExecutableStructure.get())
    , m_function(function)
    , m_constructor(constructor)
    , m_implementationVisibility(static_cast<unsigned>(implementationVisibility))
{
}

const DOMJIT::Signature* NativeExecutable::signatureFor(CodeSpecializationKind kind) const
{
    ASSERT(hasJITCodeFor(kind));
    return generatedJITCodeFor(kind)->signature();
}

Intrinsic NativeExecutable::intrinsic() const
{
    return generatedJITCodeFor(CodeForCall)->intrinsic();
}

CodeBlockHash NativeExecutable::hashFor(CodeSpecializationKind kind) const
{
    if (kind == CodeForCall)
        return CodeBlockHash(std::bit_cast<uintptr_t>(m_function));

    RELEASE_ASSERT(kind == CodeForConstruct);
    return CodeBlockHash(std::bit_cast<uintptr_t>(m_constructor));
}

JSString* NativeExecutable::toStringSlow(JSGlobalObject *globalObject)
{
    VM& vm = getVM(globalObject);

    auto throwScope = DECLARE_THROW_SCOPE(vm);

    JSValue value = jsMakeNontrivialString(globalObject, "function "_s, name(), "() {\n    [native code]\n}"_s);

    RETURN_IF_EXCEPTION(throwScope, nullptr);

    JSString* asString = ::JSC::asString(value);
    WTF::storeStoreFence();
    m_asString.set(vm, this, asString);
    return asString;
}

template<typename Visitor>
void NativeExecutable::visitChildrenImpl(JSCell* cell, Visitor& visitor)
{
    NativeExecutable* thisObject = jsCast<NativeExecutable*>(cell);
    ASSERT_GC_OBJECT_INHERITS(thisObject, info());
    Base::visitChildren(thisObject, visitor);
    visitor.append(thisObject->m_asString);
}

DEFINE_VISIT_CHILDREN(NativeExecutable);

} // namespace JSC
