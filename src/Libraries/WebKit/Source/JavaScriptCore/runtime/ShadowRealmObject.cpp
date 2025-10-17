/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 21, 2024.
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
#include "ShadowRealmObject.h"

#include "AuxiliaryBarrierInlines.h"
#include "GlobalObjectMethodTable.h"
#include "JSObjectInlines.h"
#include "StructureInlines.h"

namespace JSC {

STATIC_ASSERT_IS_TRIVIALLY_DESTRUCTIBLE(ShadowRealmObject);

} // namespace JSC

namespace JSC {

const ClassInfo ShadowRealmObject::s_info = { "ShadowRealm"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(ShadowRealmObject) };

ShadowRealmObject::ShadowRealmObject(VM& vm, Structure* structure)
    : Base(vm, structure)
{
}

template<typename Visitor>
void ShadowRealmObject::visitChildrenImpl(JSCell* cell, Visitor& visitor)
{
    ShadowRealmObject* thisObject = jsCast<ShadowRealmObject*>(cell);
    ASSERT_GC_OBJECT_INHERITS(thisObject, info());
    Base::visitChildren(thisObject, visitor);

    visitor.append(thisObject->m_globalObject);
}

DEFINE_VISIT_CHILDREN(ShadowRealmObject);

ShadowRealmObject* ShadowRealmObject::create(VM& vm, Structure* structure, JSGlobalObject* globalObject)
{
    ShadowRealmObject* object = new (NotNull, allocateCell<ShadowRealmObject>(vm)) ShadowRealmObject(vm, structure);
    object->finishCreation(vm);
    object->m_globalObject.set(vm, object, globalObject->globalObjectMethodTable()->deriveShadowRealmGlobalObject(globalObject));
    return object;
}

void ShadowRealmObject::finishCreation(VM& vm)
{
    Base::finishCreation(vm);
    ASSERT(inherits(info()));
    JSC_TO_STRING_TAG_WITHOUT_TRANSITION();
}

} // namespace JSC
