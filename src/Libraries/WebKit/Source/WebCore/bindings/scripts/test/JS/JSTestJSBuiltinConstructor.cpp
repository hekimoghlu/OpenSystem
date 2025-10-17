/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 20, 2022.
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
#include "JSTestJSBuiltinConstructor.h"

#include "ExtendedDOMClientIsoSubspaces.h"
#include "ExtendedDOMIsoSubspaces.h"
#include "JSDOMAttribute.h"
#include "JSDOMBinding.h"
#include "JSDOMBuiltinConstructor.h"
#include "JSDOMExceptionHandling.h"
#include "JSDOMGlobalObjectInlines.h"
#include "JSDOMOperation.h"
#include "JSDOMWrapperCache.h"
#include "TestJSBuiltinConstructorBuiltins.h"
#include "WebCoreJSClientData.h"
#include <JavaScriptCore/FunctionPrototype.h>
#include <JavaScriptCore/JSCInlines.h>
#include <JavaScriptCore/JSDestructibleObjectHeapCellType.h>
#include <JavaScriptCore/SlotVisitorMacros.h>
#include <JavaScriptCore/SubspaceInlines.h>
#include <wtf/GetPtr.h>
#include <wtf/PointerPreparations.h>

namespace WebCore {
using namespace JSC;

// Functions

static JSC_DECLARE_HOST_FUNCTION(jsTestJSBuiltinConstructorPrototypeFunction_testCustomFunction);

// Attributes

static JSC_DECLARE_CUSTOM_GETTER(jsTestJSBuiltinConstructorConstructor);
static JSC_DECLARE_CUSTOM_GETTER(jsTestJSBuiltinConstructor_testAttributeCustom);
static JSC_DECLARE_CUSTOM_GETTER(jsTestJSBuiltinConstructor_testAttributeRWCustom);
static JSC_DECLARE_CUSTOM_SETTER(setJSTestJSBuiltinConstructor_testAttributeRWCustom);

class JSTestJSBuiltinConstructorPrototype final : public JSC::JSNonFinalObject {
public:
    using Base = JSC::JSNonFinalObject;
    static JSTestJSBuiltinConstructorPrototype* create(JSC::VM& vm, JSDOMGlobalObject* globalObject, JSC::Structure* structure)
    {
        JSTestJSBuiltinConstructorPrototype* ptr = new (NotNull, JSC::allocateCell<JSTestJSBuiltinConstructorPrototype>(vm)) JSTestJSBuiltinConstructorPrototype(vm, globalObject, structure);
        ptr->finishCreation(vm);
        return ptr;
    }

    DECLARE_INFO;
    template<typename CellType, JSC::SubspaceAccess>
    static JSC::GCClient::IsoSubspace* subspaceFor(JSC::VM& vm)
    {
        STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(JSTestJSBuiltinConstructorPrototype, Base);
        return &vm.plainObjectSpace();
    }
    static JSC::Structure* createStructure(JSC::VM& vm, JSC::JSGlobalObject* globalObject, JSC::JSValue prototype)
    {
        return JSC::Structure::create(vm, globalObject, prototype, JSC::TypeInfo(JSC::ObjectType, StructureFlags), info());
    }

private:
    JSTestJSBuiltinConstructorPrototype(JSC::VM& vm, JSC::JSGlobalObject*, JSC::Structure* structure)
        : JSC::JSNonFinalObject(vm, structure)
    {
    }

    void finishCreation(JSC::VM&);
};
STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(JSTestJSBuiltinConstructorPrototype, JSTestJSBuiltinConstructorPrototype::Base);

using JSTestJSBuiltinConstructorDOMConstructor = JSDOMBuiltinConstructor<JSTestJSBuiltinConstructor>;

template<> const ClassInfo JSTestJSBuiltinConstructorDOMConstructor::s_info = { "TestJSBuiltinConstructor"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSTestJSBuiltinConstructorDOMConstructor) };

template<> JSValue JSTestJSBuiltinConstructorDOMConstructor::prototypeForStructure(JSC::VM& vm, const JSDOMGlobalObject& globalObject)
{
    UNUSED_PARAM(vm);
    return globalObject.functionPrototype();
}

template<> void JSTestJSBuiltinConstructorDOMConstructor::initializeProperties(VM& vm, JSDOMGlobalObject& globalObject)
{
    putDirect(vm, vm.propertyNames->length, jsNumber(0), JSC::PropertyAttribute::ReadOnly | JSC::PropertyAttribute::DontEnum);
    JSString* nameString = jsNontrivialString(vm, "TestJSBuiltinConstructor"_s);
    m_originalName.set(vm, this, nameString);
    putDirect(vm, vm.propertyNames->name, nameString, JSC::PropertyAttribute::ReadOnly | JSC::PropertyAttribute::DontEnum);
    putDirect(vm, vm.propertyNames->prototype, JSTestJSBuiltinConstructor::prototype(vm, globalObject), JSC::PropertyAttribute::ReadOnly | JSC::PropertyAttribute::DontEnum | JSC::PropertyAttribute::DontDelete);
}

template<> FunctionExecutable* JSTestJSBuiltinConstructorDOMConstructor::initializeExecutable(VM& vm)
{
    return testJSBuiltinConstructorInitializeTestJSBuiltinConstructorCodeGenerator(vm);
}

/* Hash table for prototype */

static const std::array<HashTableValue, 6> JSTestJSBuiltinConstructorPrototypeTableValues {
    HashTableValue { "constructor"_s, static_cast<unsigned>(PropertyAttribute::DontEnum), NoIntrinsic, { HashTableValue::GetterSetterType, jsTestJSBuiltinConstructorConstructor, 0 } },
    HashTableValue { "testAttribute"_s, JSC::PropertyAttribute::Accessor | JSC::PropertyAttribute::Builtin, NoIntrinsic, { HashTableValue::BuiltinAccessorType, testJSBuiltinConstructorTestAttributeCodeGenerator, setTestJSBuiltinConstructorTestAttributeCodeGenerator } },
    HashTableValue { "testAttributeCustom"_s, JSC::PropertyAttribute::ReadOnly | JSC::PropertyAttribute::CustomAccessor | JSC::PropertyAttribute::DOMAttribute, NoIntrinsic, { HashTableValue::GetterSetterType, jsTestJSBuiltinConstructor_testAttributeCustom, 0 } },
    HashTableValue { "testAttributeRWCustom"_s, JSC::PropertyAttribute::CustomAccessor | JSC::PropertyAttribute::DOMAttribute, NoIntrinsic, { HashTableValue::GetterSetterType, jsTestJSBuiltinConstructor_testAttributeRWCustom, setJSTestJSBuiltinConstructor_testAttributeRWCustom } },
    HashTableValue { "testFunction"_s, static_cast<unsigned>(JSC::PropertyAttribute::Builtin), NoIntrinsic, { HashTableValue::BuiltinGeneratorType, testJSBuiltinConstructorTestFunctionCodeGenerator, 0 } },
    HashTableValue { "testCustomFunction"_s, static_cast<unsigned>(JSC::PropertyAttribute::Function), NoIntrinsic, { HashTableValue::NativeFunctionType, jsTestJSBuiltinConstructorPrototypeFunction_testCustomFunction, 0 } },
};

const ClassInfo JSTestJSBuiltinConstructorPrototype::s_info = { "TestJSBuiltinConstructor"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSTestJSBuiltinConstructorPrototype) };

void JSTestJSBuiltinConstructorPrototype::finishCreation(VM& vm)
{
    Base::finishCreation(vm);
    reifyStaticProperties(vm, JSTestJSBuiltinConstructor::info(), JSTestJSBuiltinConstructorPrototypeTableValues, *this);
    JSC_TO_STRING_TAG_WITHOUT_TRANSITION();
}

const ClassInfo JSTestJSBuiltinConstructor::s_info = { "TestJSBuiltinConstructor"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSTestJSBuiltinConstructor) };

JSTestJSBuiltinConstructor::JSTestJSBuiltinConstructor(Structure* structure, JSDOMGlobalObject& globalObject)
    : JSDOMObject(structure, globalObject) { }

JSObject* JSTestJSBuiltinConstructor::createPrototype(VM& vm, JSDOMGlobalObject& globalObject)
{
    auto* structure = JSTestJSBuiltinConstructorPrototype::createStructure(vm, &globalObject, globalObject.objectPrototype());
    structure->setMayBePrototype(true);
    return JSTestJSBuiltinConstructorPrototype::create(vm, &globalObject, structure);
}

JSObject* JSTestJSBuiltinConstructor::prototype(VM& vm, JSDOMGlobalObject& globalObject)
{
    return getDOMPrototype<JSTestJSBuiltinConstructor>(vm, globalObject);
}

JSValue JSTestJSBuiltinConstructor::getConstructor(VM& vm, const JSGlobalObject* globalObject)
{
    return getDOMConstructor<JSTestJSBuiltinConstructorDOMConstructor, DOMConstructorID::TestJSBuiltinConstructor>(vm, *jsCast<const JSDOMGlobalObject*>(globalObject));
}

void JSTestJSBuiltinConstructor::destroy(JSC::JSCell* cell)
{
    JSTestJSBuiltinConstructor* thisObject = static_cast<JSTestJSBuiltinConstructor*>(cell);
    thisObject->JSTestJSBuiltinConstructor::~JSTestJSBuiltinConstructor();
}

JSC_DEFINE_CUSTOM_GETTER(jsTestJSBuiltinConstructorConstructor, (JSGlobalObject* lexicalGlobalObject, EncodedJSValue thisValue, PropertyName))
{
    SUPPRESS_UNCOUNTED_LOCAL auto& vm = JSC::getVM(lexicalGlobalObject);
    auto throwScope = DECLARE_THROW_SCOPE(vm);
    auto* prototype = jsDynamicCast<JSTestJSBuiltinConstructorPrototype*>(JSValue::decode(thisValue));
    if (UNLIKELY(!prototype))
        return throwVMTypeError(lexicalGlobalObject, throwScope);
    return JSValue::encode(JSTestJSBuiltinConstructor::getConstructor(vm, prototype->globalObject()));
}

static inline JSValue jsTestJSBuiltinConstructor_testAttributeCustomGetter(JSGlobalObject& lexicalGlobalObject, JSTestJSBuiltinConstructor& thisObject)
{
    UNUSED_PARAM(lexicalGlobalObject);
    return thisObject.testAttributeCustom(lexicalGlobalObject);
}

JSC_DEFINE_CUSTOM_GETTER(jsTestJSBuiltinConstructor_testAttributeCustom, (JSGlobalObject* lexicalGlobalObject, EncodedJSValue thisValue, PropertyName attributeName))
{
    return IDLAttribute<JSTestJSBuiltinConstructor>::get<jsTestJSBuiltinConstructor_testAttributeCustomGetter, CastedThisErrorBehavior::Assert>(*lexicalGlobalObject, thisValue, attributeName);
}

static inline JSValue jsTestJSBuiltinConstructor_testAttributeRWCustomGetter(JSGlobalObject& lexicalGlobalObject, JSTestJSBuiltinConstructor& thisObject)
{
    UNUSED_PARAM(lexicalGlobalObject);
    return thisObject.testAttributeRWCustom(lexicalGlobalObject);
}

JSC_DEFINE_CUSTOM_GETTER(jsTestJSBuiltinConstructor_testAttributeRWCustom, (JSGlobalObject* lexicalGlobalObject, EncodedJSValue thisValue, PropertyName attributeName))
{
    return IDLAttribute<JSTestJSBuiltinConstructor>::get<jsTestJSBuiltinConstructor_testAttributeRWCustomGetter, CastedThisErrorBehavior::Assert>(*lexicalGlobalObject, thisValue, attributeName);
}

static inline bool setJSTestJSBuiltinConstructor_testAttributeRWCustomSetter(JSGlobalObject& lexicalGlobalObject, JSTestJSBuiltinConstructor& thisObject, JSValue value)
{
    SUPPRESS_UNCOUNTED_LOCAL auto& vm = JSC::getVM(&lexicalGlobalObject);
    UNUSED_PARAM(vm);
    thisObject.setTestAttributeRWCustom(lexicalGlobalObject, value);
    return true;
}

JSC_DEFINE_CUSTOM_SETTER(setJSTestJSBuiltinConstructor_testAttributeRWCustom, (JSGlobalObject* lexicalGlobalObject, EncodedJSValue thisValue, EncodedJSValue encodedValue, PropertyName attributeName))
{
    return IDLAttribute<JSTestJSBuiltinConstructor>::set<setJSTestJSBuiltinConstructor_testAttributeRWCustomSetter>(*lexicalGlobalObject, thisValue, encodedValue, attributeName);
}

static inline JSC::EncodedJSValue jsTestJSBuiltinConstructorPrototypeFunction_testCustomFunctionBody(JSC::JSGlobalObject* lexicalGlobalObject, JSC::CallFrame* callFrame, typename IDLOperation<JSTestJSBuiltinConstructor>::ClassParameter castedThis)
{
    SUPPRESS_UNCOUNTED_LOCAL auto& vm = JSC::getVM(lexicalGlobalObject);
    auto throwScope = DECLARE_THROW_SCOPE(vm);
    UNUSED_PARAM(throwScope);
    UNUSED_PARAM(callFrame);
    RELEASE_AND_RETURN(throwScope, (JSValue::encode(castedThis->testCustomFunction(*lexicalGlobalObject, *callFrame))));
}

JSC_DEFINE_HOST_FUNCTION(jsTestJSBuiltinConstructorPrototypeFunction_testCustomFunction, (JSGlobalObject* lexicalGlobalObject, CallFrame* callFrame))
{
    return IDLOperation<JSTestJSBuiltinConstructor>::call<jsTestJSBuiltinConstructorPrototypeFunction_testCustomFunctionBody>(*lexicalGlobalObject, *callFrame, "testCustomFunction");
}

JSC::GCClient::IsoSubspace* JSTestJSBuiltinConstructor::subspaceForImpl(JSC::VM& vm)
{
    return WebCore::subspaceForImpl<JSTestJSBuiltinConstructor, UseCustomHeapCellType::No>(vm,
        [] (auto& spaces) { return spaces.m_clientSubspaceForTestJSBuiltinConstructor.get(); },
        [] (auto& spaces, auto&& space) { spaces.m_clientSubspaceForTestJSBuiltinConstructor = std::forward<decltype(space)>(space); },
        [] (auto& spaces) { return spaces.m_subspaceForTestJSBuiltinConstructor.get(); },
        [] (auto& spaces, auto&& space) { spaces.m_subspaceForTestJSBuiltinConstructor = std::forward<decltype(space)>(space); }
    );
}


}
