/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 21, 2024.
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
#include "JSIteratorConstructor.h"

#include "AbstractSlotVisitor.h"
#include "BuiltinNames.h"
#include "GetterSetter.h"
#include "JSCInlines.h"
#include "JSIterator.h"
#include "JSIteratorPrototype.h"
#include "SlotVisitor.h"

namespace JSC {

const ClassInfo JSIteratorConstructor::s_info = { "Function"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSIteratorConstructor) };

Structure* JSIteratorConstructor::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(InternalFunctionType, StructureFlags), info());
}

JSIteratorConstructor* JSIteratorConstructor::create(VM& vm, JSGlobalObject* globalObject, Structure* structure, JSIteratorPrototype* iteratorPrototype)
{
    JSIteratorConstructor* constructor = new (NotNull, allocateCell<JSIteratorConstructor>(vm)) JSIteratorConstructor(vm, structure);
    constructor->finishCreation(vm, globalObject, iteratorPrototype);
    return constructor;
}

void JSIteratorConstructor::finishCreation(VM& vm, JSGlobalObject* globalObject, JSIteratorPrototype* iteratorPrototype)
{
    Base::finishCreation(vm, 0, vm.propertyNames->Iterator.string(), PropertyAdditionMode::WithoutStructureTransition);
    putDirectWithoutTransition(vm, vm.propertyNames->prototype, iteratorPrototype, static_cast<unsigned>(PropertyAttribute::DontEnum | PropertyAttribute::DontDelete | PropertyAttribute::ReadOnly));
    JSC_BUILTIN_FUNCTION_WITHOUT_TRANSITION(vm.propertyNames->from, jsIteratorConstructorFromCodeGenerator, static_cast<unsigned>(PropertyAttribute::DontEnum));

    if (Options::useIteratorHelpers() && Options::useIteratorSequencing())
        JSC_BUILTIN_FUNCTION_WITHOUT_TRANSITION(vm.propertyNames->builtinNames().concatPublicName(), jsIteratorConstructorConcatCodeGenerator, static_cast<unsigned>(PropertyAttribute::DontEnum));
}

template<typename Visitor>
void JSIteratorConstructor::visitChildrenImpl(JSCell* cell, Visitor& visitor)
{
    auto* thisObject = jsCast<JSIteratorConstructor*>(cell);
    ASSERT_GC_OBJECT_INHERITS(thisObject, info());
    Base::visitChildren(thisObject, visitor);
}

DEFINE_VISIT_CHILDREN(JSIteratorConstructor);

static JSC_DECLARE_HOST_FUNCTION(callIterator);
static JSC_DECLARE_HOST_FUNCTION(constructIterator);

JSIteratorConstructor::JSIteratorConstructor(VM& vm, Structure* structure)
    : Base(vm, structure, callIterator, constructIterator)
{
}

JSC_DEFINE_HOST_FUNCTION(callIterator, (JSGlobalObject* globalObject, CallFrame*))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    return JSValue::encode(throwConstructorCannotBeCalledAsFunctionTypeError(globalObject, scope, "Iterator"_s));
}

// https://tc39.es/proposal-iterator-helpers/#sec-iterator
JSC_DEFINE_HOST_FUNCTION(constructIterator, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSObject* newTarget = asObject(callFrame->newTarget());
    JSIteratorConstructor* iteratorConstructor = jsCast<JSIteratorConstructor*>(callFrame->jsCallee());
    if (newTarget == iteratorConstructor)
        return JSValue::encode(throwTypeError(globalObject, scope, "Iterator cannot be constructed directly"_s));

    Structure* iteratorStructure = JSC_GET_DERIVED_STRUCTURE(vm, iteratorStructure, newTarget, callFrame->jsCallee());
    RETURN_IF_EXCEPTION(scope, { });

    RELEASE_AND_RETURN(scope, JSValue::encode(JSIterator::create(vm, iteratorStructure)));
}

} // namespace JSC
