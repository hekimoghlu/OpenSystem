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
#include "config.h"
#include "JSDOMBuiltinConstructorBase.h"

#include "WebCoreJSClientData.h"
#include <JavaScriptCore/JSCInlines.h>

namespace WebCore {
using namespace JSC;

template<typename Visitor>
void JSDOMBuiltinConstructorBase::visitChildrenImpl(JSC::JSCell* cell, Visitor& visitor)
{
    auto* thisObject = jsCast<JSDOMBuiltinConstructorBase*>(cell);
    ASSERT_GC_OBJECT_INHERITS(thisObject, info());
    Base::visitChildren(thisObject, visitor);
    visitor.append(thisObject->m_initializeFunction);
}

DEFINE_VISIT_CHILDREN(JSDOMBuiltinConstructorBase);

JSC::GCClient::IsoSubspace* JSDOMBuiltinConstructorBase::subspaceForImpl(JSC::VM& vm)
{
    return &static_cast<JSVMClientData*>(vm.clientData)->domBuiltinConstructorSpace();
}

} // namespace WebCore
