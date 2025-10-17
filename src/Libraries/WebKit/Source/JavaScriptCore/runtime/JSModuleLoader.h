/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 18, 2025.
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

#include "JSObject.h"

namespace JSC {

class JSInternalPromise;
class JSModuleNamespaceObject;
class JSModuleRecord;
class SourceCode;

class JSModuleLoader final : public JSNonFinalObject {
public:
    using Base = JSNonFinalObject;
    static constexpr unsigned StructureFlags = Base::StructureFlags;

    template<typename CellType, SubspaceAccess>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(JSModuleLoader, Base);
        return &vm.plainObjectSpace();
    }

    enum Status {
        Fetch = 1,
        Instantiate,
        Satisfy,
        Link,
        Ready,
    };

    static JSModuleLoader* create(JSGlobalObject* globalObject, VM& vm, Structure* structure)
    {
        JSModuleLoader* object = new (NotNull, allocateCell<JSModuleLoader>(vm)) JSModuleLoader(vm, structure);
        object->finishCreation(globalObject, vm);
        return object;
    }

    DECLARE_INFO;

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    // APIs to control the module loader.
    JSValue provideFetch(JSGlobalObject*, JSValue key, const SourceCode&);
    JSInternalPromise* loadAndEvaluateModule(JSGlobalObject*, JSValue moduleName, JSValue parameters, JSValue scriptFetcher);
    JSInternalPromise* loadModule(JSGlobalObject*, JSValue moduleName, JSValue parameters, JSValue scriptFetcher);
    JSValue linkAndEvaluateModule(JSGlobalObject*, JSValue moduleKey, JSValue scriptFetcher);
    JSInternalPromise* requestImportModule(JSGlobalObject*, const Identifier&, JSValue referrer, JSValue parameters, JSValue scriptFetcher);

    // Platform dependent hooked APIs.
    JSInternalPromise* importModule(JSGlobalObject*, JSString* moduleName, JSValue parameters, const SourceOrigin& referrer);
    Identifier resolve(JSGlobalObject*, JSValue name, JSValue referrer, JSValue scriptFetcher);
    JSInternalPromise* fetch(JSGlobalObject*, JSValue key, JSValue parameters, JSValue scriptFetcher);
    JSObject* createImportMetaProperties(JSGlobalObject*, JSValue key, JSModuleRecord*, JSValue scriptFetcher);

    // Additional platform dependent hooked APIs.
    JSValue evaluate(JSGlobalObject*, JSValue key, JSValue moduleRecord, JSValue scriptFetcher, JSValue sentValue, JSValue resumeMode);
    JSValue evaluateNonVirtual(JSGlobalObject*, JSValue key, JSValue moduleRecord, JSValue scriptFetcher, JSValue sentValue, JSValue resumeMode);

    // Utility functions.
    JSModuleNamespaceObject* getModuleNamespaceObject(JSGlobalObject*, JSValue moduleRecord);
    JSArray* dependencyKeysIfEvaluated(JSGlobalObject*, JSValue key);

private:
    JSModuleLoader(VM&, Structure*);
    void finishCreation(JSGlobalObject*, VM&);
};

} // namespace JSC
