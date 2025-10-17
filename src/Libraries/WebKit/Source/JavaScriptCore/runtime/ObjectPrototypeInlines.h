/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 13, 2024.
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

#include "ArrayPrototypeInlines.h"
#include "IntegrityInlines.h"
#include "ObjectPrototype.h"

namespace JSC {

#if PLATFORM(IOS) || PLATFORM(VISION)
bool isPokerBros();
#endif

inline std::tuple<ASCIILiteral, JSString*> inferBuiltinTag(JSGlobalObject* globalObject, JSObject* object)
{
    VM& vm = getVM(globalObject);
    auto scope = DECLARE_THROW_SCOPE(vm);

#if PLATFORM(IOS) || PLATFORM(VISION)
    static bool needsOldBuiltinTag = isPokerBros();
    if (UNLIKELY(needsOldBuiltinTag))
        return std::tuple { object->className(), nullptr };
#endif

    switch (object->type()) {
    case ArrayType:
    case DerivedArrayType:
        return std::tuple { "Array"_s, vm.smallStrings.objectArrayString() };
    case DirectArgumentsType:
    case ScopedArgumentsType:
    case ClonedArgumentsType:
        return std::tuple { "Arguments"_s, vm.smallStrings.objectArgumentsString() };
    case JSFunctionType:
    case InternalFunctionType:
        return std::tuple { "Function"_s, vm.smallStrings.objectFunctionString() };
    case ErrorInstanceType:
        return std::tuple { "Error"_s, vm.smallStrings.objectErrorString() };
    case JSDateType:
        return std::tuple { "Date"_s, vm.smallStrings.objectDateString() };
    case RegExpObjectType:
        return std::tuple { "RegExp"_s, vm.smallStrings.objectRegExpString() };
    case BooleanObjectType:
        return std::tuple { "Boolean"_s, vm.smallStrings.objectBooleanString() };
    case NumberObjectType:
        return std::tuple { "Number"_s, vm.smallStrings.objectNumberString() };
    case StringObjectType:
    case DerivedStringObjectType:
        return std::tuple { "String"_s, vm.smallStrings.objectStringString() };
    case FinalObjectType:
        return std::tuple { "Object"_s, vm.smallStrings.objectObjectString() };
    default: {
        bool objectIsArray = isArray(globalObject, object);
        RETURN_IF_EXCEPTION(scope, { });
        if (objectIsArray)
            return std::tuple { "Array"_s, vm.smallStrings.objectArrayString() };
        if (object->isCallable())
            return std::tuple { "Function"_s, vm.smallStrings.objectFunctionString() };
        return std::tuple { "Object"_s, vm.smallStrings.objectObjectString() };
    }
    }
}

ALWAYS_INLINE JSString* objectPrototypeToStringSlow(JSGlobalObject* globalObject, JSObject* thisObject)
{
    VM& vm = getVM(globalObject);
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto [ tag, jsCommonTag ] = inferBuiltinTag(globalObject, thisObject);
    RETURN_IF_EXCEPTION(scope, nullptr);
    JSString* jsTag = nullptr;

    PropertySlot slot(thisObject, PropertySlot::InternalMethodType::Get);
    bool hasProperty = thisObject->getPropertySlot(globalObject, vm.propertyNames->toStringTagSymbol, slot);
    EXCEPTION_ASSERT(!scope.exception() || !hasProperty);
    if (hasProperty) {
        JSValue tagValue = slot.getValue(globalObject, vm.propertyNames->toStringTagSymbol);
        RETURN_IF_EXCEPTION(scope, nullptr);
        if (tagValue.isString())
            jsTag = asString(tagValue);
    }

    JSString* jsResult = nullptr;
    if (!jsTag) {
        if (jsCommonTag)
            jsResult = jsCommonTag;
        else
            jsTag = jsString(vm, AtomStringImpl::add(tag));
    }

    if (!jsResult) {
        jsResult = jsString(globalObject, vm.smallStrings.objectStringStart(), jsTag, vm.smallStrings.singleCharacterString(']'));
        RETURN_IF_EXCEPTION(scope, nullptr);
    }

    thisObject->structure()->cacheSpecialProperty(globalObject, vm, jsResult, CachedSpecialPropertyKey::ToStringTag, slot);
    return jsResult;
}

ALWAYS_INLINE JSString* objectPrototypeToString(JSGlobalObject* globalObject, JSValue thisValue)
{
    VM& vm = getVM(globalObject);
    auto scope = DECLARE_THROW_SCOPE(vm);

    if (thisValue.isUndefined())
        return vm.smallStrings.objectUndefinedString();
    if (thisValue.isNull())
        return vm.smallStrings.objectNullString();

    JSObject* thisObject = thisValue.toObject(globalObject);
    RETURN_IF_EXCEPTION(scope, nullptr);

    Integrity::auditStructureID(thisObject->structureID());
    auto result = thisObject->structure()->cachedSpecialProperty(CachedSpecialPropertyKey::ToStringTag);
    if (result)
        return asString(result);

    RELEASE_AND_RETURN(scope, objectPrototypeToStringSlow(globalObject, thisObject));
}

inline Structure* ObjectPrototype::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(ObjectType, StructureFlags), info());
}

} // namespace JSC
