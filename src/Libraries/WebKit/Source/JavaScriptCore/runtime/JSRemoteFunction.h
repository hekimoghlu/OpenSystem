/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 21, 2025.
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

#include "AuxiliaryBarrier.h"
#include "JSFunction.h"
#include "JSObject.h"
#include <wtf/TaggedArrayStoragePtr.h>

namespace JSC {

JSC_DECLARE_HOST_FUNCTION(remoteFunctionCallForJSFunction);
JSC_DECLARE_HOST_FUNCTION(remoteFunctionCallGeneric);
JSC_DECLARE_HOST_FUNCTION(isRemoteFunction);
JSC_DECLARE_HOST_FUNCTION(createRemoteFunction);

// JSRemoteFunction creates a bridge between its native Realm and a remote one.
// When invoked, arguments are wrapped to prevent leaking information across the realm boundary.
// The return value and any abrupt completions are also filtered.
class JSRemoteFunction final : public JSFunction {
public:
    using Base = JSFunction;
    static constexpr unsigned StructureFlags = Base::StructureFlags;

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.remoteFunctionSpace<mode>();
    }

    JS_EXPORT_PRIVATE static JSRemoteFunction* tryCreate(JSGlobalObject*, VM&, JSObject* targetCallable);

    JSObject* targetFunction() { return m_targetFunction.get(); }
    JSGlobalObject* targetGlobalObject() { return targetFunction()->globalObject(); }
    JSString* nameMayBeNull() const { return m_nameMayBeNull.get(); }
    String nameString()
    {
        if (!m_nameMayBeNull)
            return emptyString();
        ASSERT(!m_nameMayBeNull->isRope());
        bool allocationAllowed = false;
        return m_nameMayBeNull->tryGetValue(allocationAllowed);
    }

    double length(VM&) const { return m_length; }

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);
    
    static constexpr ptrdiff_t offsetOfTargetFunction() { return OBJECT_OFFSETOF(JSRemoteFunction, m_targetFunction); }

    DECLARE_EXPORT_INFO;

private:
    JSRemoteFunction(VM&, NativeExecutable*, JSGlobalObject*, Structure*, JSObject* targetCallable);

    void copyNameAndLength(JSGlobalObject*);

    void finishCreation(JSGlobalObject*, VM&);
    DECLARE_VISIT_CHILDREN;

    WriteBarrier<JSObject> m_targetFunction;
    WriteBarrier<JSString> m_nameMayBeNull;
    double m_length;
};

}
