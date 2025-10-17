/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 20, 2024.
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

#include "JSGlobalObject.h"

OBJC_CLASS JSScript;

namespace JSC {

class JSAPIGlobalObject final : public JSGlobalObject {
public:
    using Base = JSGlobalObject;

    DECLARE_EXPORT_INFO;

    static constexpr DestructionMode needsDestruction = NeedsDestruction;
    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.apiGlobalObjectSpace<mode>();
    }

    static JSAPIGlobalObject* create(VM&, Structure*);
    static Structure* createStructure(VM&, JSValue prototype);

    static void reportUncaughtExceptionAtEventLoop(JSGlobalObject*, Exception*);

    JSValue loadAndEvaluateJSScriptModule(const JSLockHolder&, JSScript *);

private:
    static const GlobalObjectMethodTable* globalObjectMethodTable();
    JSAPIGlobalObject(VM&, Structure*);

    static JSInternalPromise* moduleLoaderImportModule(JSGlobalObject*, JSModuleLoader*, JSString* moduleNameValue, JSValue parameters, const SourceOrigin&);
    static Identifier moduleLoaderResolve(JSGlobalObject*, JSModuleLoader*, JSValue keyValue, JSValue referrerValue, JSValue);
    static JSInternalPromise* moduleLoaderFetch(JSGlobalObject*, JSModuleLoader*, JSValue, JSValue, JSValue);
    static JSObject* moduleLoaderCreateImportMetaProperties(JSGlobalObject*, JSModuleLoader*, JSValue, JSModuleRecord*, JSValue);
    static JSValue moduleLoaderEvaluate(JSGlobalObject*, JSModuleLoader*, JSValue, JSValue, JSValue, JSValue, JSValue);
};

}
