/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 19, 2025.
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

#include "JSDOMConvert.h"
#include <JavaScriptCore/JSIteratorPrototype.h>
#include <JavaScriptCore/PropertySlot.h>
#include <type_traits>

namespace WebCore {

void addValueIterableMethods(JSC::JSGlobalObject&, JSC::JSObject&);

enum class JSDOMIteratorType { Set, Map };

// struct IteratorTraits {
//     static constexpr JSDOMIteratorType type = [Map|Set];
//     using KeyType = [IDLType|void];
//     using ValueType = [IDLType];
// };

template<typename T, typename U = void> using EnableIfMap = typename std::enable_if<T::type == JSDOMIteratorType::Map, U>::type;
template<typename T, typename U = void> using EnableIfSet = typename std::enable_if<T::type == JSDOMIteratorType::Set, U>::type;

template<typename JSWrapper, typename IteratorTraits> class JSDOMIteratorPrototype final : public JSC::JSNonFinalObject {
public:
    using Base = JSC::JSNonFinalObject;
    using DOMWrapped = typename JSWrapper::DOMWrapped;

    template<typename CellType, JSC::SubspaceAccess>
    static JSC::GCClient::IsoSubspace* subspaceFor(JSC::VM& vm)
    {
        STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(JSDOMIteratorPrototype, Base);
        return &vm.plainObjectSpace();
    }

    static JSDOMIteratorPrototype* create(JSC::VM& vm, JSC::JSGlobalObject* globalObject, JSC::Structure* structure)
    {
        STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(JSDOMIteratorPrototype, JSDOMIteratorPrototype::Base);
        JSDOMIteratorPrototype* prototype = new (NotNull, JSC::allocateCell<JSDOMIteratorPrototype>(vm)) JSDOMIteratorPrototype(vm, structure);
        prototype->finishCreation(vm, globalObject);
        return prototype;
    }

    DECLARE_INFO;

    static JSC::Structure* createStructure(JSC::VM& vm, JSC::JSGlobalObject* globalObject, JSC::JSValue prototype)
    {
        return JSC::Structure::create(vm, globalObject, prototype, JSC::TypeInfo(JSC::ObjectType, StructureFlags), info());
    }

    static JSC::EncodedJSValue JSC_HOST_CALL_ATTRIBUTES next(JSC::JSGlobalObject*, JSC::CallFrame*);

private:
    JSDOMIteratorPrototype(JSC::VM& vm, JSC::Structure* structure) : Base(vm, structure) { }

    void finishCreation(JSC::VM&, JSC::JSGlobalObject*);
};

using IterationKind = JSC::IterationKind;

template<typename JSWrapper, typename IteratorTraits> class JSDOMIteratorBase : public JSDOMObject {
public:
    using Base = JSDOMObject;

    using Wrapper = JSWrapper;
    using Traits = IteratorTraits;

    using DOMWrapped = typename Wrapper::DOMWrapped;
    using Prototype = JSDOMIteratorPrototype<Wrapper, Traits>;

    DECLARE_INFO;

    static Prototype* createPrototype(JSC::VM& vm, JSC::JSGlobalObject& globalObject)
    {
        auto* structure = Prototype::createStructure(vm, &globalObject, globalObject.iteratorPrototype());
        structure->setMayBePrototype(true);
        return Prototype::create(vm, &globalObject, structure);
    }

    JSC::JSValue next(JSC::JSGlobalObject&);

    static void createStructure(JSC::VM&, JSC::JSGlobalObject*, JSC::JSValue); // Make use of createStructure for this compile-error.

protected:
    JSDOMIteratorBase(JSC::Structure* structure, JSWrapper& iteratedObject, IterationKind kind)
        : Base(structure, *iteratedObject.globalObject())
        , m_iterator(iteratedObject.wrapped().createIterator(iteratedObject.globalObject()->scriptExecutionContext()))
        , m_kind(kind)
    {
    }

    template<typename IteratorValue, typename T = Traits> EnableIfMap<T, JSC::JSValue> asJS(JSC::JSGlobalObject&, IteratorValue&);
    template<typename IteratorValue, typename T = Traits> EnableIfSet<T, JSC::JSValue> asJS(JSC::JSGlobalObject&, IteratorValue&);

    static void destroy(JSC::JSCell*);

    std::optional<typename DOMWrapped::Iterator> m_iterator;
    IterationKind m_kind;
};

inline JSC::JSValue jsPair(JSC::JSGlobalObject&, JSDOMGlobalObject& globalObject, JSC::JSValue value1, JSC::JSValue value2)
{
    JSC::MarkedArgumentBuffer arguments;
    arguments.append(value1);
    arguments.append(value2);
    ASSERT(!arguments.hasOverflowed());
    return constructArray(&globalObject, static_cast<JSC::ArrayAllocationProfile*>(nullptr), arguments);
}

template<typename FirstType, typename SecondType, typename T, typename U> 
inline JSC::JSValue jsPair(JSC::JSGlobalObject& lexicalGlobalObject, JSDOMGlobalObject& globalObject, const T& value1, const U& value2)
{
    return jsPair(lexicalGlobalObject, globalObject, toJS<FirstType>(lexicalGlobalObject, globalObject, value1), toJS<SecondType>(lexicalGlobalObject, globalObject, value2));
}

template<typename JSIterator> JSC::JSValue iteratorCreate(typename JSIterator::Wrapper&, IterationKind);
template<typename JSIterator> JSC::JSValue iteratorForEach(JSC::JSGlobalObject&, JSC::CallFrame&, typename JSIterator::Wrapper&);

template<typename JSIterator> JSC::JSValue iteratorCreate(typename JSIterator::Wrapper& thisObject, IterationKind kind)
{
    ASSERT(thisObject.globalObject());
    JSDOMGlobalObject& globalObject = *thisObject.globalObject();
    return JSIterator::create(globalObject.vm(), getDOMStructure<JSIterator>(globalObject.vm(), globalObject), thisObject, kind);
}

template<typename JSWrapper, typename IteratorTraits>
template<typename IteratorValue, typename T> inline EnableIfMap<T, JSC::JSValue> JSDOMIteratorBase<JSWrapper, IteratorTraits>::asJS(JSC::JSGlobalObject& lexicalGlobalObject, IteratorValue& value)
{
    ASSERT(value);
    
    switch (m_kind) {
    case IterationKind::Keys:
        return toJS<typename Traits::KeyType>(lexicalGlobalObject, *globalObject(), value->key);
    case IterationKind::Values:
        return toJS<typename Traits::ValueType>(lexicalGlobalObject, *globalObject(), value->value);
    case IterationKind::Entries:
        return jsPair<typename Traits::KeyType, typename Traits::ValueType>(lexicalGlobalObject, *globalObject(), value->key, value->value);
    };
    
    ASSERT_NOT_REACHED();
    return { };
}

template<typename JSWrapper, typename IteratorTraits>
template<typename IteratorValue, typename T> inline EnableIfSet<T, JSC::JSValue> JSDOMIteratorBase<JSWrapper, IteratorTraits>::asJS(JSC::JSGlobalObject& lexicalGlobalObject, IteratorValue& value)
{
    ASSERT(value);

    auto globalObject = this->globalObject();
    auto result = toJS<typename Traits::ValueType>(lexicalGlobalObject, *globalObject, value);

    switch (m_kind) {
    case IterationKind::Keys:
    case IterationKind::Values:
        return result;
    case IterationKind::Entries:
        return jsPair(lexicalGlobalObject, *globalObject, result, result);
    };

    ASSERT_NOT_REACHED();
    return { };
}

template<typename JSIterator, typename IteratorValue> EnableIfMap<typename JSIterator::Traits> appendForEachArguments(JSC::JSGlobalObject& lexicalGlobalObject, JSDOMGlobalObject& globalObject, JSC::MarkedArgumentBuffer& arguments, IteratorValue& value)
{
    ASSERT(value);
    arguments.append(toJS<typename JSIterator::Traits::ValueType>(lexicalGlobalObject, globalObject, value->value));
    arguments.append(toJS<typename JSIterator::Traits::KeyType>(lexicalGlobalObject, globalObject, value->key));
}

template<typename JSIterator, typename IteratorValue> EnableIfSet<typename JSIterator::Traits> appendForEachArguments(JSC::JSGlobalObject& lexicalGlobalObject, JSDOMGlobalObject& globalObject, JSC::MarkedArgumentBuffer& arguments, IteratorValue& value)
{
    ASSERT(value);
    auto argument = toJS<typename JSIterator::Traits::ValueType>(lexicalGlobalObject, globalObject, value);
    arguments.append(argument);
    arguments.append(argument);
}

template<typename JSIterator> JSC::JSValue iteratorForEach(JSC::JSGlobalObject& lexicalGlobalObject, JSC::CallFrame& callFrame, typename JSIterator::Wrapper& thisObject)
{
    auto& vm = JSC::getVM(&lexicalGlobalObject);
    auto scope = DECLARE_THROW_SCOPE(vm);
    JSC::JSValue callback = callFrame.argument(0);
    JSC::JSValue thisValue = callFrame.argument(1);

    auto callData = JSC::getCallData(callback);
    if (callData.type == JSC::CallData::Type::None)
        return throwTypeError(&lexicalGlobalObject, scope, "Cannot call callback"_s);

    auto iterator = thisObject.wrapped().createIterator(JSC::jsCast<JSDOMGlobalObject*>(&lexicalGlobalObject)->scriptExecutionContext());
    while (auto value = iterator.next()) {
        JSC::MarkedArgumentBuffer arguments;
        appendForEachArguments<JSIterator>(lexicalGlobalObject, *thisObject.globalObject(), arguments, value);
        arguments.append(&thisObject);
        if (UNLIKELY(arguments.hasOverflowed())) {
            throwOutOfMemoryError(&lexicalGlobalObject, scope);
            return { };
        }
        JSC::call(&lexicalGlobalObject, callback, callData, thisValue, arguments);
        if (UNLIKELY(scope.exception()))
            break;
    }
    return JSC::jsUndefined();
}

template<typename JSWrapper, typename IteratorTraits>
void JSDOMIteratorBase<JSWrapper, IteratorTraits>::destroy(JSCell* cell)
{
    JSDOMIteratorBase<JSWrapper, IteratorTraits>* thisObject = static_cast<JSDOMIteratorBase<JSWrapper, IteratorTraits>*>(cell);
    thisObject->JSDOMIteratorBase<JSWrapper, IteratorTraits>::~JSDOMIteratorBase();
}

template<typename JSWrapper, typename IteratorTraits>
JSC::JSValue JSDOMIteratorBase<JSWrapper, IteratorTraits>::next(JSC::JSGlobalObject& lexicalGlobalObject)
{
    if (m_iterator) {
        auto iteratorValue = m_iterator->next();
        if (iteratorValue)
            return createIteratorResultObject(&lexicalGlobalObject, asJS(lexicalGlobalObject, iteratorValue), false);
        m_iterator = std::nullopt;
    }
    return createIteratorResultObject(&lexicalGlobalObject, JSC::jsUndefined(), true);
}

template<typename JSWrapper, typename IteratorTraits>
JSC::EncodedJSValue JSC_HOST_CALL_ATTRIBUTES JSDOMIteratorPrototype<JSWrapper, IteratorTraits>::next(JSC::JSGlobalObject* globalObject, JSC::CallFrame* callFrame)
{
    JSC::VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto iterator = JSC::jsDynamicCast<JSDOMIteratorBase<JSWrapper, IteratorTraits>*>(callFrame->thisValue());
    if (!iterator)
        return JSC::JSValue::encode(throwTypeError(globalObject, scope, "Cannot call next() on a non-Iterator object"_s));

    return JSC::JSValue::encode(iterator->next(*globalObject));
}

template<typename JSWrapper, typename IteratorTraits>
void JSDOMIteratorPrototype<JSWrapper, IteratorTraits>::finishCreation(JSC::VM& vm, JSC::JSGlobalObject* globalObject)
{
    Base::finishCreation(vm);
    ASSERT(inherits(info()));

    JSC_NATIVE_FUNCTION_WITHOUT_TRANSITION(vm.propertyNames->next, next, 0, 0, JSC::ImplementationVisibility::Public);
    JSC_TO_STRING_TAG_WITHOUT_TRANSITION();
}

}
