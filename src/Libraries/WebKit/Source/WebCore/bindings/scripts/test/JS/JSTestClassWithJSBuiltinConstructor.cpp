/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 12, 2024.
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
#include "JSTestClassWithJSBuiltinConstructor.h"

#include "ActiveDOMObject.h"
#include "ExtendedDOMClientIsoSubspaces.h"
#include "ExtendedDOMIsoSubspaces.h"
#include "JSDOMBinding.h"
#include "JSDOMBuiltinConstructor.h"
#include "JSDOMExceptionHandling.h"
#include "JSDOMGlobalObjectInlines.h"
#include "JSDOMWrapperCache.h"
#include "ScriptExecutionContext.h"
#include "TestClassWithJSBuiltinConstructorBuiltins.h"
#include "WebCoreJSClientData.h"
#include <JavaScriptCore/FunctionPrototype.h>
#include <JavaScriptCore/HeapAnalyzer.h>
#include <JavaScriptCore/JSCInlines.h>
#include <JavaScriptCore/JSDestructibleObjectHeapCellType.h>
#include <JavaScriptCore/SlotVisitorMacros.h>
#include <JavaScriptCore/SubspaceInlines.h>
#include <wtf/GetPtr.h>
#include <wtf/PointerPreparations.h>
#include <wtf/URL.h>
#include <wtf/text/MakeString.h>

namespace WebCore {
using namespace JSC;

// Attributes

static JSC_DECLARE_CUSTOM_GETTER(jsTestClassWithJSBuiltinConstructorConstructor);

class JSTestClassWithJSBuiltinConstructorPrototype final : public JSC::JSNonFinalObject {
public:
    using Base = JSC::JSNonFinalObject;
    static JSTestClassWithJSBuiltinConstructorPrototype* create(JSC::VM& vm, JSDOMGlobalObject* globalObject, JSC::Structure* structure)
    {
        JSTestClassWithJSBuiltinConstructorPrototype* ptr = new (NotNull, JSC::allocateCell<JSTestClassWithJSBuiltinConstructorPrototype>(vm)) JSTestClassWithJSBuiltinConstructorPrototype(vm, globalObject, structure);
        ptr->finishCreation(vm);
        return ptr;
    }

    DECLARE_INFO;
    template<typename CellType, JSC::SubspaceAccess>
    static JSC::GCClient::IsoSubspace* subspaceFor(JSC::VM& vm)
    {
        STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(JSTestClassWithJSBuiltinConstructorPrototype, Base);
        return &vm.plainObjectSpace();
    }
    static JSC::Structure* createStructure(JSC::VM& vm, JSC::JSGlobalObject* globalObject, JSC::JSValue prototype)
    {
        return JSC::Structure::create(vm, globalObject, prototype, JSC::TypeInfo(JSC::ObjectType, StructureFlags), info());
    }

private:
    JSTestClassWithJSBuiltinConstructorPrototype(JSC::VM& vm, JSC::JSGlobalObject*, JSC::Structure* structure)
        : JSC::JSNonFinalObject(vm, structure)
    {
    }

    void finishCreation(JSC::VM&);
};
STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(JSTestClassWithJSBuiltinConstructorPrototype, JSTestClassWithJSBuiltinConstructorPrototype::Base);

using JSTestClassWithJSBuiltinConstructorDOMConstructor = JSDOMBuiltinConstructor<JSTestClassWithJSBuiltinConstructor>;

template<> const ClassInfo JSTestClassWithJSBuiltinConstructorDOMConstructor::s_info = { "TestClassWithJSBuiltinConstructor"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSTestClassWithJSBuiltinConstructorDOMConstructor) };

template<> JSValue JSTestClassWithJSBuiltinConstructorDOMConstructor::prototypeForStructure(JSC::VM& vm, const JSDOMGlobalObject& globalObject)
{
    UNUSED_PARAM(vm);
    return globalObject.functionPrototype();
}

template<> void JSTestClassWithJSBuiltinConstructorDOMConstructor::initializeProperties(VM& vm, JSDOMGlobalObject& globalObject)
{
    putDirect(vm, vm.propertyNames->length, jsNumber(0), JSC::PropertyAttribute::ReadOnly | JSC::PropertyAttribute::DontEnum);
    JSString* nameString = jsNontrivialString(vm, "TestClassWithJSBuiltinConstructor"_s);
    m_originalName.set(vm, this, nameString);
    putDirect(vm, vm.propertyNames->name, nameString, JSC::PropertyAttribute::ReadOnly | JSC::PropertyAttribute::DontEnum);
    putDirect(vm, vm.propertyNames->prototype, JSTestClassWithJSBuiltinConstructor::prototype(vm, globalObject), JSC::PropertyAttribute::ReadOnly | JSC::PropertyAttribute::DontEnum | JSC::PropertyAttribute::DontDelete);
}

template<> FunctionExecutable* JSTestClassWithJSBuiltinConstructorDOMConstructor::initializeExecutable(VM& vm)
{
    return testClassWithJSBuiltinConstructorInitializeTestClassWithJSBuiltinConstructorCodeGenerator(vm);
}

/* Hash table for prototype */

static const std::array<HashTableValue, 1> JSTestClassWithJSBuiltinConstructorPrototypeTableValues {
    HashTableValue { "constructor"_s, static_cast<unsigned>(PropertyAttribute::DontEnum), NoIntrinsic, { HashTableValue::GetterSetterType, jsTestClassWithJSBuiltinConstructorConstructor, 0 } },
};

const ClassInfo JSTestClassWithJSBuiltinConstructorPrototype::s_info = { "TestClassWithJSBuiltinConstructor"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSTestClassWithJSBuiltinConstructorPrototype) };

void JSTestClassWithJSBuiltinConstructorPrototype::finishCreation(VM& vm)
{
    Base::finishCreation(vm);
    reifyStaticProperties(vm, JSTestClassWithJSBuiltinConstructor::info(), JSTestClassWithJSBuiltinConstructorPrototypeTableValues, *this);
    JSC_TO_STRING_TAG_WITHOUT_TRANSITION();
}

const ClassInfo JSTestClassWithJSBuiltinConstructor::s_info = { "TestClassWithJSBuiltinConstructor"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSTestClassWithJSBuiltinConstructor) };

JSTestClassWithJSBuiltinConstructor::JSTestClassWithJSBuiltinConstructor(Structure* structure, JSDOMGlobalObject& globalObject, Ref<TestClassWithJSBuiltinConstructor>&& impl)
    : JSDOMWrapper<TestClassWithJSBuiltinConstructor>(structure, globalObject, WTFMove(impl))
{
}

static_assert(!std::is_base_of<ActiveDOMObject, TestClassWithJSBuiltinConstructor>::value, "Interface is not marked as [ActiveDOMObject] even though implementation class subclasses ActiveDOMObject.");

JSObject* JSTestClassWithJSBuiltinConstructor::createPrototype(VM& vm, JSDOMGlobalObject& globalObject)
{
    auto* structure = JSTestClassWithJSBuiltinConstructorPrototype::createStructure(vm, &globalObject, globalObject.objectPrototype());
    structure->setMayBePrototype(true);
    return JSTestClassWithJSBuiltinConstructorPrototype::create(vm, &globalObject, structure);
}

JSObject* JSTestClassWithJSBuiltinConstructor::prototype(VM& vm, JSDOMGlobalObject& globalObject)
{
    return getDOMPrototype<JSTestClassWithJSBuiltinConstructor>(vm, globalObject);
}

JSValue JSTestClassWithJSBuiltinConstructor::getConstructor(VM& vm, const JSGlobalObject* globalObject)
{
    return getDOMConstructor<JSTestClassWithJSBuiltinConstructorDOMConstructor, DOMConstructorID::TestClassWithJSBuiltinConstructor>(vm, *jsCast<const JSDOMGlobalObject*>(globalObject));
}

void JSTestClassWithJSBuiltinConstructor::destroy(JSC::JSCell* cell)
{
    JSTestClassWithJSBuiltinConstructor* thisObject = static_cast<JSTestClassWithJSBuiltinConstructor*>(cell);
    thisObject->JSTestClassWithJSBuiltinConstructor::~JSTestClassWithJSBuiltinConstructor();
}

JSC_DEFINE_CUSTOM_GETTER(jsTestClassWithJSBuiltinConstructorConstructor, (JSGlobalObject* lexicalGlobalObject, EncodedJSValue thisValue, PropertyName))
{
    SUPPRESS_UNCOUNTED_LOCAL auto& vm = JSC::getVM(lexicalGlobalObject);
    auto throwScope = DECLARE_THROW_SCOPE(vm);
    auto* prototype = jsDynamicCast<JSTestClassWithJSBuiltinConstructorPrototype*>(JSValue::decode(thisValue));
    if (UNLIKELY(!prototype))
        return throwVMTypeError(lexicalGlobalObject, throwScope);
    return JSValue::encode(JSTestClassWithJSBuiltinConstructor::getConstructor(vm, prototype->globalObject()));
}

JSC::GCClient::IsoSubspace* JSTestClassWithJSBuiltinConstructor::subspaceForImpl(JSC::VM& vm)
{
    return WebCore::subspaceForImpl<JSTestClassWithJSBuiltinConstructor, UseCustomHeapCellType::No>(vm,
        [] (auto& spaces) { return spaces.m_clientSubspaceForTestClassWithJSBuiltinConstructor.get(); },
        [] (auto& spaces, auto&& space) { spaces.m_clientSubspaceForTestClassWithJSBuiltinConstructor = std::forward<decltype(space)>(space); },
        [] (auto& spaces) { return spaces.m_subspaceForTestClassWithJSBuiltinConstructor.get(); },
        [] (auto& spaces, auto&& space) { spaces.m_subspaceForTestClassWithJSBuiltinConstructor = std::forward<decltype(space)>(space); }
    );
}

void JSTestClassWithJSBuiltinConstructor::analyzeHeap(JSCell* cell, HeapAnalyzer& analyzer)
{
    auto* thisObject = jsCast<JSTestClassWithJSBuiltinConstructor*>(cell);
    analyzer.setWrappedObjectForCell(cell, &thisObject->wrapped());
    if (RefPtr context = thisObject->scriptExecutionContext())
        analyzer.setLabelForCell(cell, makeString("url "_s, context->url().string()));
    Base::analyzeHeap(cell, analyzer);
}

bool JSTestClassWithJSBuiltinConstructorOwner::isReachableFromOpaqueRoots(JSC::Handle<JSC::Unknown> handle, void*, AbstractSlotVisitor& visitor, ASCIILiteral* reason)
{
    UNUSED_PARAM(handle);
    UNUSED_PARAM(visitor);
    UNUSED_PARAM(reason);
    return false;
}

void JSTestClassWithJSBuiltinConstructorOwner::finalize(JSC::Handle<JSC::Unknown> handle, void* context)
{
    auto* jsTestClassWithJSBuiltinConstructor = static_cast<JSTestClassWithJSBuiltinConstructor*>(handle.slot()->asCell());
    auto& world = *static_cast<DOMWrapperWorld*>(context);
    uncacheWrapper(world, jsTestClassWithJSBuiltinConstructor->protectedWrapped().ptr(), jsTestClassWithJSBuiltinConstructor);
}

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
#if ENABLE(BINDING_INTEGRITY)
#if PLATFORM(WIN)
#pragma warning(disable: 4483)
extern "C" { extern void (*const __identifier("??_7TestClassWithJSBuiltinConstructor@WebCore@@6B@")[])(); }
#else
extern "C" { extern void* _ZTVN7WebCore33TestClassWithJSBuiltinConstructorE[]; }
#endif
template<typename T, typename = std::enable_if_t<std::is_same_v<T, TestClassWithJSBuiltinConstructor>, void>> static inline void verifyVTable(TestClassWithJSBuiltinConstructor* ptr) {
    if constexpr (std::is_polymorphic_v<T>) {
        const void* actualVTablePointer = getVTablePointer<T>(ptr);
#if PLATFORM(WIN)
        void* expectedVTablePointer = __identifier("??_7TestClassWithJSBuiltinConstructor@WebCore@@6B@");
#else
        void* expectedVTablePointer = &_ZTVN7WebCore33TestClassWithJSBuiltinConstructorE[2];
#endif

        // If you hit this assertion you either have a use after free bug, or
        // TestClassWithJSBuiltinConstructor has subclasses. If TestClassWithJSBuiltinConstructor has subclasses that get passed
        // to toJS() we currently require TestClassWithJSBuiltinConstructor you to opt out of binding hardening
        // by adding the SkipVTableValidation attribute to the interface IDL definition
        RELEASE_ASSERT(actualVTablePointer == expectedVTablePointer);
    }
}
#endif
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

JSC::JSValue toJSNewlyCreated(JSC::JSGlobalObject*, JSDOMGlobalObject* globalObject, Ref<TestClassWithJSBuiltinConstructor>&& impl)
{
#if ENABLE(BINDING_INTEGRITY)
    verifyVTable<TestClassWithJSBuiltinConstructor>(impl.ptr());
#endif
    return createWrapper<TestClassWithJSBuiltinConstructor>(globalObject, WTFMove(impl));
}

JSC::JSValue toJS(JSC::JSGlobalObject* lexicalGlobalObject, JSDOMGlobalObject* globalObject, TestClassWithJSBuiltinConstructor& impl)
{
    return wrap(lexicalGlobalObject, globalObject, impl);
}

TestClassWithJSBuiltinConstructor* JSTestClassWithJSBuiltinConstructor::toWrapped(JSC::VM&, JSC::JSValue value)
{
    if (auto* wrapper = jsDynamicCast<JSTestClassWithJSBuiltinConstructor*>(value))
        return &wrapper->wrapped();
    return nullptr;
}

}
