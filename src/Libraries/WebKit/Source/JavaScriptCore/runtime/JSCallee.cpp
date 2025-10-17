/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 23, 2022.
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
#include "JSCallee.h"

#include "JSCInlines.h"

namespace JSC {

const ClassInfo JSCallee::s_info = { "Callee"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSCallee) };

JSCallee::JSCallee(VM& vm, JSGlobalObject* globalObject, Structure* structure)
    : Base(vm, structure)
    , m_scope(globalObject, WriteBarrierEarlyInit)
{
}

JSCallee::JSCallee(VM& vm, JSScope* scope, Structure* structure)
    : Base(vm, structure)
    , m_scope(scope, WriteBarrierEarlyInit)
{
}

template<typename Visitor>
void JSCallee::visitChildrenImpl(JSCell* cell, Visitor& visitor)
{
    JSCallee* thisObject = jsCast<JSCallee*>(cell);
    ASSERT_GC_OBJECT_INHERITS(thisObject, info());
    Base::visitChildren(thisObject, visitor);

    visitor.append(thisObject->m_scope);
}

DEFINE_VISIT_CHILDREN(JSCallee);

} // namespace JSC
