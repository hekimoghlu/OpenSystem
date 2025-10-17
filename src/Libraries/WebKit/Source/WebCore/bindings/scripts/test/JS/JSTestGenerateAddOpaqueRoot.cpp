/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 20, 2021.
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
#include "JSTestGenerateAddOpaqueRoot.h"

#include "ActiveDOMObject.h"
#include "ExtendedDOMClientIsoSubspaces.h"
#include "ExtendedDOMIsoSubspaces.h"
#include "JSDOMAttribute.h"
#include "JSDOMBinding.h"
#include "JSDOMConstructorNotConstructable.h"
#include "JSDOMConvertStrings.h"
#include "JSDOMExceptionHandling.h"
#include "JSDOMGlobalObjectInlines.h"
#include "JSDOMWrapperCache.h"
#include "ScriptExecutionContext.h"
#include "WebCoreJSClientData.h"
#include "WebCoreOpaqueRootInlines.h"
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

static JSC_DECLARE_CUSTOM_GETTER(jsTestGenerateAddOpaqueRootConstructor);
static JSC_DECLARE_CUSTOM_GETTER(jsTestGenerateAddOpaqueRoot_someAttribute);

class JSTestGenerateAddOpaqueRootPrototype final : public JSC::JSNonFinalObject {
public:
    using Base = JSC::JSNonFinalObject;
    static JSTestGenerateAddOpaqueRootPrototype* create(JSC::VM& vm, JSDOMGlobalObject* globalObject, JSC::Structure* structure)
    {
        JSTestGenerateAddOpaqueRootPrototype* ptr = new (NotNull, JSC::allocateCell<JSTestGenerateAddOpaqueRootPrototype>(vm)) JSTestGenerateAddOpaqueRootPrototype(vm, globalObject, structure);
        ptr->finishCreation(vm);
        return ptr;
    }

    DECLARE_INFO;
    template<typename CellType, JSC::SubspaceAccess>
    static JSC::GCClient::IsoSubspace* subspaceFor(JSC::VM& vm)
    {
        STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(JSTestGenerateAddOpaqueRootPrototype, Base);
        return &vm.plainObjectSpace();
    }
    static JSC::Structure* createStructure(JSC::VM& vm, JSC::JSGlobalObject* globalObject, JSC::JSValue prototype)
    {
        return JSC::Structure::create(vm, globalObject, prototype, JSC::TypeInfo(JSC::ObjectType, StructureFlags), info());
    }

private:
    JSTestGenerateAddOpaqueRootPrototype(JSC::VM& vm, JSC::JSGlobalObject*, JSC::Structure* structure)
        : JSC::JSNonFinalObject(vm, structure)
    {
    }

    void finishCreation(JSC::VM&);
};
STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(JSTestGenerateAddOpaqueRootPrototype, JSTestGenerateAddOpaqueRootPrototype::Base);

using JSTestGenerateAddOpaqueRootDOMConstructor = JSDOMConstructorNotConstructable<JSTestGenerateAddOpaqueRoot>;

template<> const ClassInfo JSTestGenerateAddOpaqueRootDOMConstructor::s_info = { "TestGenerateAddOpaqueRoot"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSTestGenerateAddOpaqueRootDOMConstructor) };

template<> JSValue JSTestGenerateAddOpaqueRootDOMConstructor::prototypeForStructure(JSC::VM& vm, const JSDOMGlobalObject& globalObject)
{
    UNUSED_PARAM(vm);
    return globalObject.functionPrototype();
}

template<> void JSTestGenerateAddOpaqueRootDOMConstructor::initializeProperties(VM& vm, JSDOMGlobalObject& globalObject)
{
    putDirect(vm, vm.propertyNames->length, jsNumber(0), JSC::PropertyAttribute::ReadOnly | JSC::PropertyAttribute::DontEnum);
    JSString* nameString = jsNontrivialString(vm, "TestGenerateAddOpaqueRoot"_s);
    m_originalName.set(vm, this, nameString);
    putDirect(vm, vm.propertyNames->name, nameString, JSC::PropertyAttribute::ReadOnly | JSC::PropertyAttribute::DontEnum);
    putDirect(vm, vm.propertyNames->prototype, JSTestGenerateAddOpaqueRoot::prototype(vm, globalObject), JSC::PropertyAttribute::ReadOnly | JSC::PropertyAttribute::DontEnum | JSC::PropertyAttribute::DontDelete);
}

/* Hash table for prototype */

static const std::array<HashTableValue, 2> JSTestGenerateAddOpaqueRootPrototypeTableValues {
    HashTableValue { "constructor"_s, static_cast<unsigned>(PropertyAttribute::DontEnum), NoIntrinsic, { HashTableValue::GetterSetterType, jsTestGenerateAddOpaqueRootConstructor, 0 } },
    HashTableValue { "someAttribute"_s, JSC::PropertyAttribute::ReadOnly | JSC::PropertyAttribute::CustomAccessor | JSC::PropertyAttribute::DOMAttribute, NoIntrinsic, { HashTableValue::GetterSetterType, jsTestGenerateAddOpaqueRoot_someAttribute, 0 } },
};

const ClassInfo JSTestGenerateAddOpaqueRootPrototype::s_info = { "TestGenerateAddOpaqueRoot"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSTestGenerateAddOpaqueRootPrototype) };

void JSTestGenerateAddOpaqueRootPrototype::finishCreation(VM& vm)
{
    Base::finishCreation(vm);
    reifyStaticProperties(vm, JSTestGenerateAddOpaqueRoot::info(), JSTestGenerateAddOpaqueRootPrototypeTableValues, *this);
    JSC_TO_STRING_TAG_WITHOUT_TRANSITION();
}

const ClassInfo JSTestGenerateAddOpaqueRoot::s_info = { "TestGenerateAddOpaqueRoot"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSTestGenerateAddOpaqueRoot) };

JSTestGenerateAddOpaqueRoot::JSTestGenerateAddOpaqueRoot(Structure* structure, JSDOMGlobalObject& globalObject, Ref<TestGenerateAddOpaqueRoot>&& impl)
    : JSDOMWrapper<TestGenerateAddOpaqueRoot>(structure, globalObject, WTFMove(impl))
{
}

static_assert(!std::is_base_of<ActiveDOMObject, TestGenerateAddOpaqueRoot>::value, "Interface is not marked as [ActiveDOMObject] even though implementation class subclasses ActiveDOMObject.");

JSObject* JSTestGenerateAddOpaqueRoot::createPrototype(VM& vm, JSDOMGlobalObject& globalObject)
{
    auto* structure = JSTestGenerateAddOpaqueRootPrototype::createStructure(vm, &globalObject, globalObject.objectPrototype());
    structure->setMayBePrototype(true);
    return JSTestGenerateAddOpaqueRootPrototype::create(vm, &globalObject, structure);
}

JSObject* JSTestGenerateAddOpaqueRoot::prototype(VM& vm, JSDOMGlobalObject& globalObject)
{
    return getDOMPrototype<JSTestGenerateAddOpaqueRoot>(vm, globalObject);
}

JSValue JSTestGenerateAddOpaqueRoot::getConstructor(VM& vm, const JSGlobalObject* globalObject)
{
    return getDOMConstructor<JSTestGenerateAddOpaqueRootDOMConstructor, DOMConstructorID::TestGenerateAddOpaqueRoot>(vm, *jsCast<const JSDOMGlobalObject*>(globalObject));
}

void JSTestGenerateAddOpaqueRoot::destroy(JSC::JSCell* cell)
{
    JSTestGenerateAddOpaqueRoot* thisObject = static_cast<JSTestGenerateAddOpaqueRoot*>(cell);
    thisObject->JSTestGenerateAddOpaqueRoot::~JSTestGenerateAddOpaqueRoot();
}

JSC_DEFINE_CUSTOM_GETTER(jsTestGenerateAddOpaqueRootConstructor, (JSGlobalObject* lexicalGlobalObject, EncodedJSValue thisValue, PropertyName))
{
    SUPPRESS_UNCOUNTED_LOCAL auto& vm = JSC::getVM(lexicalGlobalObject);
    auto throwScope = DECLARE_THROW_SCOPE(vm);
    auto* prototype = jsDynamicCast<JSTestGenerateAddOpaqueRootPrototype*>(JSValue::decode(thisValue));
    if (UNLIKELY(!prototype))
        return throwVMTypeError(lexicalGlobalObject, throwScope);
    return JSValue::encode(JSTestGenerateAddOpaqueRoot::getConstructor(vm, prototype->globalObject()));
}

static inline JSValue jsTestGenerateAddOpaqueRoot_someAttributeGetter(JSGlobalObject& lexicalGlobalObject, JSTestGenerateAddOpaqueRoot& thisObject)
{
    SUPPRESS_UNCOUNTED_LOCAL auto& vm = JSC::getVM(&lexicalGlobalObject);
    auto throwScope = DECLARE_THROW_SCOPE(vm);
    SUPPRESS_UNCOUNTED_LOCAL auto& impl = thisObject.wrapped();
    RELEASE_AND_RETURN(throwScope, (toJS<IDLDOMString>(lexicalGlobalObject, throwScope, impl.someAttribute())));
}

JSC_DEFINE_CUSTOM_GETTER(jsTestGenerateAddOpaqueRoot_someAttribute, (JSGlobalObject* lexicalGlobalObject, EncodedJSValue thisValue, PropertyName attributeName))
{
    return IDLAttribute<JSTestGenerateAddOpaqueRoot>::get<jsTestGenerateAddOpaqueRoot_someAttributeGetter, CastedThisErrorBehavior::Assert>(*lexicalGlobalObject, thisValue, attributeName);
}

JSC::GCClient::IsoSubspace* JSTestGenerateAddOpaqueRoot::subspaceForImpl(JSC::VM& vm)
{
    return WebCore::subspaceForImpl<JSTestGenerateAddOpaqueRoot, UseCustomHeapCellType::No>(vm,
        [] (auto& spaces) { return spaces.m_clientSubspaceForTestGenerateAddOpaqueRoot.get(); },
        [] (auto& spaces, auto&& space) { spaces.m_clientSubspaceForTestGenerateAddOpaqueRoot = std::forward<decltype(space)>(space); },
        [] (auto& spaces) { return spaces.m_subspaceForTestGenerateAddOpaqueRoot.get(); },
        [] (auto& spaces, auto&& space) { spaces.m_subspaceForTestGenerateAddOpaqueRoot = std::forward<decltype(space)>(space); }
    );
}

template<typename Visitor>
void JSTestGenerateAddOpaqueRoot::visitChildrenImpl(JSCell* cell, Visitor& visitor)
{
    auto* thisObject = jsCast<JSTestGenerateAddOpaqueRoot*>(cell);
    ASSERT_GC_OBJECT_INHERITS(thisObject, info());
    Base::visitChildren(thisObject, visitor);
    addWebCoreOpaqueRoot(visitor, thisObject->wrapped().ownerObjectConcurrently());
}

DEFINE_VISIT_CHILDREN(JSTestGenerateAddOpaqueRoot);

void JSTestGenerateAddOpaqueRoot::analyzeHeap(JSCell* cell, HeapAnalyzer& analyzer)
{
    auto* thisObject = jsCast<JSTestGenerateAddOpaqueRoot*>(cell);
    analyzer.setWrappedObjectForCell(cell, &thisObject->wrapped());
    if (RefPtr context = thisObject->scriptExecutionContext())
        analyzer.setLabelForCell(cell, makeString("url "_s, context->url().string()));
    Base::analyzeHeap(cell, analyzer);
}

bool JSTestGenerateAddOpaqueRootOwner::isReachableFromOpaqueRoots(JSC::Handle<JSC::Unknown> handle, void*, AbstractSlotVisitor& visitor, ASCIILiteral* reason)
{
    UNUSED_PARAM(handle);
    UNUSED_PARAM(visitor);
    UNUSED_PARAM(reason);
    return false;
}

void JSTestGenerateAddOpaqueRootOwner::finalize(JSC::Handle<JSC::Unknown> handle, void* context)
{
    auto* jsTestGenerateAddOpaqueRoot = static_cast<JSTestGenerateAddOpaqueRoot*>(handle.slot()->asCell());
    auto& world = *static_cast<DOMWrapperWorld*>(context);
    uncacheWrapper(world, jsTestGenerateAddOpaqueRoot->protectedWrapped().ptr(), jsTestGenerateAddOpaqueRoot);
}

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
#if ENABLE(BINDING_INTEGRITY)
#if PLATFORM(WIN)
#pragma warning(disable: 4483)
extern "C" { extern void (*const __identifier("??_7TestGenerateAddOpaqueRoot@WebCore@@6B@")[])(); }
#else
extern "C" { extern void* _ZTVN7WebCore25TestGenerateAddOpaqueRootE[]; }
#endif
template<typename T, typename = std::enable_if_t<std::is_same_v<T, TestGenerateAddOpaqueRoot>, void>> static inline void verifyVTable(TestGenerateAddOpaqueRoot* ptr) {
    if constexpr (std::is_polymorphic_v<T>) {
        const void* actualVTablePointer = getVTablePointer<T>(ptr);
#if PLATFORM(WIN)
        void* expectedVTablePointer = __identifier("??_7TestGenerateAddOpaqueRoot@WebCore@@6B@");
#else
        void* expectedVTablePointer = &_ZTVN7WebCore25TestGenerateAddOpaqueRootE[2];
#endif

        // If you hit this assertion you either have a use after free bug, or
        // TestGenerateAddOpaqueRoot has subclasses. If TestGenerateAddOpaqueRoot has subclasses that get passed
        // to toJS() we currently require TestGenerateAddOpaqueRoot you to opt out of binding hardening
        // by adding the SkipVTableValidation attribute to the interface IDL definition
        RELEASE_ASSERT(actualVTablePointer == expectedVTablePointer);
    }
}
#endif
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

JSC::JSValue toJSNewlyCreated(JSC::JSGlobalObject*, JSDOMGlobalObject* globalObject, Ref<TestGenerateAddOpaqueRoot>&& impl)
{
#if ENABLE(BINDING_INTEGRITY)
    verifyVTable<TestGenerateAddOpaqueRoot>(impl.ptr());
#endif
    return createWrapper<TestGenerateAddOpaqueRoot>(globalObject, WTFMove(impl));
}

JSC::JSValue toJS(JSC::JSGlobalObject* lexicalGlobalObject, JSDOMGlobalObject* globalObject, TestGenerateAddOpaqueRoot& impl)
{
    return wrap(lexicalGlobalObject, globalObject, impl);
}

TestGenerateAddOpaqueRoot* JSTestGenerateAddOpaqueRoot::toWrapped(JSC::VM&, JSC::JSValue value)
{
    if (auto* wrapper = jsDynamicCast<JSTestGenerateAddOpaqueRoot*>(value))
        return &wrapper->wrapped();
    return nullptr;
}

}
