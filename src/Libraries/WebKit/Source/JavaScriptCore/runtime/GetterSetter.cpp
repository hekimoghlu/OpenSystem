/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 30, 2022.
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
#include "GetterSetter.h"

#include "Exception.h"
#include "JSObjectInlines.h"
#include <wtf/Assertions.h>

namespace JSC {

STATIC_ASSERT_IS_TRIVIALLY_DESTRUCTIBLE(GetterSetter);

const ClassInfo GetterSetter::s_info = { "GetterSetter"_s, nullptr, nullptr, nullptr, CREATE_METHOD_TABLE(GetterSetter) };

template<typename Visitor>
void GetterSetter::visitChildrenImpl(JSCell* cell, Visitor& visitor)
{
    GetterSetter* thisObject = jsCast<GetterSetter*>(cell);
    ASSERT_GC_OBJECT_INHERITS(thisObject, info());
    Base::visitChildren(thisObject, visitor);

    visitor.append(thisObject->m_getter);
    visitor.append(thisObject->m_setter);
}

DEFINE_VISIT_CHILDREN(GetterSetter);

JSValue GetterSetter::callGetter(JSGlobalObject* globalObject, JSValue thisValue)
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    // FIXME: Some callers may invoke get() without checking for an exception first.
    // We work around that by checking here.
    RETURN_IF_EXCEPTION(scope, scope.exception()->value());

    JSObject* getter = this->getter();

    auto callData = JSC::getCallData(getter);
    RELEASE_AND_RETURN(scope, call(globalObject, getter, callData, thisValue, ArgList()));
}

bool GetterSetter::callSetter(JSGlobalObject* globalObject, JSValue thisValue, JSValue value, bool shouldThrow)
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSObject* setter = this->setter();

    if (setter->type() == NullSetterFunctionType)
        return typeError(globalObject, scope, shouldThrow, ReadonlyPropertyWriteError);

    MarkedArgumentBuffer args;
    args.append(value);
    ASSERT(!args.hasOverflowed());

    auto callData = JSC::getCallData(setter);
    scope.release();
    call(globalObject, setter, callData, thisValue, args);
    return true;
}

} // namespace JSC
