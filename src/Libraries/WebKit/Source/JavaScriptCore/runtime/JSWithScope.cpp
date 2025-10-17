/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 5, 2024.
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
#include "JSWithScope.h"

#include "JSCInlines.h"

namespace JSC {

const ClassInfo JSWithScope::s_info = { "WithScope"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSWithScope) };

JSWithScope* JSWithScope::create(
    VM& vm, JSGlobalObject* globalObject, JSScope* next, JSObject* object)
{
    Structure* structure = globalObject->withScopeStructure();
    JSWithScope* withScope = new (NotNull, allocateCell<JSWithScope>(vm)) JSWithScope(vm, structure, object, next);
    withScope->finishCreation(vm);
    return withScope;
}

template<typename Visitor>
void JSWithScope::visitChildrenImpl(JSCell* cell, Visitor& visitor)
{
    JSWithScope* thisObject = jsCast<JSWithScope*>(cell);
    ASSERT_GC_OBJECT_INHERITS(thisObject, info());
    Base::visitChildren(thisObject, visitor);
    visitor.append(thisObject->m_object);
}

DEFINE_VISIT_CHILDREN(JSWithScope);

Structure* JSWithScope::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue proto)
{
    return Structure::create(vm, globalObject, proto, TypeInfo(WithScopeType, StructureFlags), info());
}

JSWithScope::JSWithScope(VM& vm, Structure* structure, JSObject* object, JSScope* next)
    : Base(vm, structure, next)
    , m_object(object, WriteBarrierEarlyInit)
{
}

} // namespace JSC
