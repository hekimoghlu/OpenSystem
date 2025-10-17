/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 28, 2024.
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
#include "JSIDBSerializationGlobalObject.h"

#include "JSDOMGlobalObjectInlines.h"
#include "WebCoreJSClientData.h"

namespace WebCore {

using namespace JSC;

const ClassInfo JSIDBSerializationGlobalObject::s_info = { "JSIDBSerializationGlobalObject"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSIDBSerializationGlobalObject) };

inline JSIDBSerializationGlobalObject::JSIDBSerializationGlobalObject(VM& vm, Structure* structure, Ref<DOMWrapperWorld>&& impl)
    : Base(vm, structure, WTFMove(impl))
    , m_scriptExecutionContext(EmptyScriptExecutionContext::create(vm))
{
}

JSIDBSerializationGlobalObject* JSIDBSerializationGlobalObject::create(VM& vm, Structure* structure, Ref<DOMWrapperWorld>&& impl)
{
    JSIDBSerializationGlobalObject* ptr =  new (NotNull, allocateCell<JSIDBSerializationGlobalObject>(vm)) JSIDBSerializationGlobalObject(vm, structure, WTFMove(impl));
    ptr->finishCreation(vm);
    return ptr;
}

void JSIDBSerializationGlobalObject::finishCreation(VM& vm)
{
    Base::finishCreation(vm);
}

GCClient::IsoSubspace* JSIDBSerializationGlobalObject::subspaceForImpl(VM& vm)
{
    return &static_cast<JSVMClientData*>(vm.clientData)->idbSerializationSpace();
}

void JSIDBSerializationGlobalObject::destroy(JSCell* cell)
{
    static_cast<JSIDBSerializationGlobalObject*>(cell)->JSIDBSerializationGlobalObject::~JSIDBSerializationGlobalObject();
}

} // namespace WebCore

