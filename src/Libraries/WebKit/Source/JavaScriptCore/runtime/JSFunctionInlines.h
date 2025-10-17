/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 29, 2021.
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

#include "ExecutableBaseInlines.h"
#include "FunctionExecutable.h"
#include "JSBoundFunction.h"
#include "JSFunction.h"
#include "JSRemoteFunction.h"
#include "NativeExecutable.h"
#include <wtf/text/MakeString.h>

namespace JSC {

inline Structure* JSFunction::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    ASSERT(globalObject);
    return Structure::create(vm, globalObject, prototype, TypeInfo(JSFunctionType, StructureFlags), info());
}

inline Structure* JSStrictFunction::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    ASSERT(globalObject);
    return Structure::create(vm, globalObject, prototype, TypeInfo(JSFunctionType, StructureFlags), info());
}

inline Structure* JSSloppyFunction::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    ASSERT(globalObject);
    return Structure::create(vm, globalObject, prototype, TypeInfo(JSFunctionType, StructureFlags), info());
}

inline Structure* JSArrowFunction::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    ASSERT(globalObject);
    return Structure::create(vm, globalObject, prototype, TypeInfo(JSFunctionType, StructureFlags), info());
}

inline JSFunction* JSFunction::createWithInvalidatedReallocationWatchpoint(VM& vm, JSGlobalObject* globalObject, FunctionExecutable* executable, JSScope* scope)
{
    return createWithInvalidatedReallocationWatchpoint(vm, globalObject, executable, scope, selectStructureForNewFuncExp(globalObject, executable));
}

inline JSFunction* JSFunction::createWithInvalidatedReallocationWatchpoint(VM& vm, JSGlobalObject*, FunctionExecutable* executable, JSScope* scope, Structure* structure)
{
    ASSERT(executable->singleton().hasBeenInvalidated());
    return createImpl(vm, executable, scope, structure);
}

inline JSFunction::JSFunction(VM& vm, FunctionExecutable* executable, JSScope* scope, Structure* structure)
    : Base(vm, scope, structure)
    , m_executableOrRareData(std::bit_cast<uintptr_t>(executable))
{
    assertTypeInfoFlagInvariants();
}

inline FunctionExecutable* JSFunction::jsExecutable() const
{
    ASSERT(!isHostFunctionNonInline());
    return static_cast<FunctionExecutable*>(executable());
}

inline bool JSFunction::isHostFunction() const
{
    ASSERT(executable());
    return executable()->isHostFunction();
}

inline bool JSFunction::isNonBoundHostFunction() const
{
    return isHostFunction() && !inherits<JSBoundFunction>();
}

inline Intrinsic JSFunction::intrinsic() const
{
    return executable()->intrinsic();
}

inline bool JSFunction::isBuiltinFunction() const
{
    return !isHostFunction() && jsExecutable()->isBuiltinFunction();
}

inline bool JSFunction::isHostOrBuiltinFunction() const
{
    return isHostFunction() || isBuiltinFunction();
}

inline bool JSFunction::isClassConstructorFunction() const
{
    return !isHostFunction() && jsExecutable()->isClassConstructorFunction();
}

inline bool JSFunction::isRemoteFunction() const
{
    return inherits<JSRemoteFunction>();
}

inline TaggedNativeFunction JSFunction::nativeFunction()
{
    ASSERT(isHostFunctionNonInline());
    return static_cast<NativeExecutable*>(executable())->function();
}

inline TaggedNativeFunction JSFunction::nativeConstructor()
{
    ASSERT(isHostFunctionNonInline());
    return static_cast<NativeExecutable*>(executable())->constructor();
}

inline bool isRemoteFunction(JSValue value)
{
    return value.inherits<JSRemoteFunction>();
}

inline bool JSFunction::hasReifiedLength() const
{
    if (FunctionRareData* rareData = this->rareData())
        return rareData->hasReifiedLength();
    return false;
}

inline bool JSFunction::hasReifiedName() const
{
    if (FunctionRareData* rareData = this->rareData())
        return rareData->hasReifiedName();
    return false;
}

inline double JSFunction::originalLength(VM& vm)
{
    if (inherits<JSBoundFunction>())
        return jsCast<JSBoundFunction*>(this)->length(vm);
    if (inherits<JSRemoteFunction>())
        return jsCast<JSRemoteFunction*>(this)->length(vm);
    ASSERT(!isHostFunction());
    return jsExecutable()->parameterCount();
}

template<typename... StringTypes>
ALWAYS_INLINE String makeNameWithOutOfMemoryCheck(JSGlobalObject* globalObject, ThrowScope& throwScope, ASCIILiteral messagePrefix, StringTypes... strings)
{
    String name = tryMakeString(strings...);
    if (UNLIKELY(!name)) {
        throwOutOfMemoryError(globalObject, throwScope, makeString(messagePrefix, "name is too long"_s));
        return String();
    }
    return name;
}

inline JSString* JSFunction::originalName(JSGlobalObject* globalObject)
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    if (this->inherits<JSBoundFunction>()) {
        JSString* nameMayBeNull = jsCast<JSBoundFunction*>(this)->nameMayBeNull();
        if (nameMayBeNull)
            RELEASE_AND_RETURN(scope, jsString(globalObject, vm.smallStrings.boundPrefixString(), nameMayBeNull));
        return jsEmptyString(vm);
    }

    if (this->inherits<JSRemoteFunction>()) {
        JSString* nameMayBeNull = jsCast<JSRemoteFunction*>(this)->nameMayBeNull();
        if (nameMayBeNull)
            return nameMayBeNull;
        return jsEmptyString(vm);
    }

    ASSERT(!isHostFunction());
    const Identifier& ecmaName = jsExecutable()->ecmaName();
    String name;
    // https://tc39.github.io/ecma262/#sec-exports-runtime-semantics-evaluation
    // When the ident is "*default*", we need to set "default" for the ecma name.
    // This "*default*" name is never shown to users.
    if (ecmaName == vm.propertyNames->starDefaultPrivateName)
        name = vm.propertyNames->defaultKeyword.string();
    else
        name = ecmaName.string();

    if (jsExecutable()->isGetter()) {
        name = makeNameWithOutOfMemoryCheck(globalObject, scope, "Getter "_s, "get "_s, name);
        RETURN_IF_EXCEPTION(scope, { });
    } else if (jsExecutable()->isSetter()) {
        name = makeNameWithOutOfMemoryCheck(globalObject, scope, "Setter "_s, "set "_s, name);
        RETURN_IF_EXCEPTION(scope, { });
    }
    return jsString(vm, WTFMove(name));
}

inline bool JSFunction::canAssumeNameAndLengthAreOriginal(VM&)
{
    // Bound functions are not eagerly generating name and length.
    // Thus, we can use FunctionRareData's tracking. This is useful to optimize func.bind().bind() case.
    if (isNonBoundHostFunction())
        return false;
    FunctionRareData* rareData = this->rareData();
    if (!rareData)
        return true;
    if (rareData->hasModifiedNameForBoundOrNonHostFunction())
        return false;
    if (rareData->hasModifiedLengthForBoundOrNonHostFunction())
        return false;
    return true;
}

inline bool JSFunction::mayHaveNonReifiedPrototype()
{
    return !isHostOrBuiltinFunction() && jsExecutable()->hasPrototypeProperty();
}

inline bool JSFunction::canUseAllocationProfiles()
{
    if (isHostOrBuiltinFunction()) {
        if (isHostFunction())
            return false;

        VM& vm = globalObject()->vm();
        unsigned attributes;
        JSValue prototype = getDirect(vm, vm.propertyNames->prototype, attributes);
        if (!prototype || (attributes & PropertyAttribute::AccessorOrCustomAccessorOrValue))
            return false;
    }

    // If we don't have a prototype property, we're not guaranteed it's
    // non-configurable. For example, user code can define the prototype
    // as a getter. JS semantics require that the getter is called every
    // time |construct| occurs with this function as new.target.
    return jsExecutable()->hasPrototypeProperty();
}

inline FunctionRareData* JSFunction::ensureRareDataAndObjectAllocationProfile(JSGlobalObject* globalObject, unsigned inlineCapacity)
{
    ASSERT(canUseAllocationProfiles());
    FunctionRareData* rareData = this->rareData();
    if (!rareData)
        return allocateAndInitializeRareData(globalObject, inlineCapacity);
    if (UNLIKELY(!rareData->isObjectAllocationProfileInitialized()))
        return initializeRareData(globalObject, inlineCapacity);
    return rareData;
}

inline JSString* JSFunction::asStringConcurrently() const
{
    if (inherits<JSBoundFunction>() || inherits<JSRemoteFunction>())
        return nullptr;
    if (isHostFunction())
        return static_cast<NativeExecutable*>(executable())->asStringConcurrently();
    return jsExecutable()->asStringConcurrently();
}

} // namespace JSC
