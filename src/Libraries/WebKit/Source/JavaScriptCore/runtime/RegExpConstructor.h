/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 27, 2024.
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

#include "InternalFunction.h"
#include "RegExp.h"
#include "RegExpCachedResult.h"
#include "RegExpObject.h"

namespace JSC {

class RegExpPrototype;
class GetterSetter;

class RegExpConstructor final : public InternalFunction {
public:
    typedef InternalFunction Base;
    static constexpr unsigned StructureFlags = Base::StructureFlags | HasStaticPropertyTable;

    static RegExpConstructor* create(VM& vm, Structure* structure, RegExpPrototype* regExpPrototype)
    {
        RegExpConstructor* constructor = new (NotNull, allocateCell<RegExpConstructor>(vm)) RegExpConstructor(vm, structure);
        constructor->finishCreation(vm, regExpPrototype);
        return constructor;
    }

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    DECLARE_INFO;

private:
    RegExpConstructor(VM&, Structure*);
    void finishCreation(VM&, RegExpPrototype*);
};
STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(RegExpConstructor, InternalFunction);

JSObject* constructRegExp(JSGlobalObject*, const ArgList&, JSObject* callee = nullptr, JSValue newTarget = JSValue());

ALWAYS_INLINE bool isRegExp(VM& vm, JSGlobalObject* globalObject, JSValue value)
{
    auto scope = DECLARE_THROW_SCOPE(vm);
    if (!value.isObject())
        return false;

    JSObject* object = asObject(value);
    JSValue matchValue = object->get(globalObject, vm.propertyNames->matchSymbol);
    RETURN_IF_EXCEPTION(scope, false);
    if (!matchValue.isUndefined())
        return matchValue.toBoolean(globalObject);

    return object->inherits<RegExpObject>();
}

JSC_DECLARE_HOST_FUNCTION(esSpecRegExpCreate);
JSC_DECLARE_HOST_FUNCTION(esSpecIsRegExp);

} // namespace JSC
