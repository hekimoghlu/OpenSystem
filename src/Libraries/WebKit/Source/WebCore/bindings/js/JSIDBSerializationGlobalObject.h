/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 11, 2023.
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

#include "EmptyScriptExecutionContext.h"
#include "JSDOMGlobalObject.h"

namespace WebCore {

class JSIDBSerializationGlobalObject final : public JSDOMGlobalObject {
public:
    using Base = JSDOMGlobalObject;
    static JSIDBSerializationGlobalObject* create(JSC::VM&, JSC::Structure*, Ref<DOMWrapperWorld>&&);

    template<typename, JSC::SubspaceAccess mode>
    static JSC::GCClient::IsoSubspace* subspaceFor(JSC::VM& vm)
    {
        if constexpr (mode == JSC::SubspaceAccess::Concurrently)
            return nullptr;
        return subspaceForImpl(vm);
    }
    static JSC::GCClient::IsoSubspace* subspaceForImpl(JSC::VM&);

    DECLARE_INFO;
    static JSC::Structure* createStructure(JSC::VM& vm, JSC::JSValue prototype)
    {
        return JSC::Structure::create(vm, 0, prototype, JSC::TypeInfo(JSC::GlobalObjectType, StructureFlags), info());
    }
    static void destroy(JSC::JSCell*);

    ScriptExecutionContext* scriptExecutionContext() const { return m_scriptExecutionContext.ptr(); }

private:
    JSIDBSerializationGlobalObject(JSC::VM&, JSC::Structure*, Ref<DOMWrapperWorld>&&);
    void finishCreation(JSC::VM&);

    Ref<EmptyScriptExecutionContext> m_scriptExecutionContext;
};

} // namespace WebCore

