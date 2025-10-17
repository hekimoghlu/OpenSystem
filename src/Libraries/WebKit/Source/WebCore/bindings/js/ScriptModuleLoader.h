/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 17, 2024.
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

#include "ModuleScriptLoader.h"
#include "ModuleScriptLoaderClient.h"
#include <JavaScriptCore/JSCJSValue.h>
#include <wtf/HashSet.h>
#include <wtf/Noncopyable.h>
#include <wtf/RobinHoodHashMap.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/URLHash.h>

namespace JSC {

class CallFrame;
class JSGlobalObject;
class JSInternalPromise;
class JSModuleLoader;
class JSModuleRecord;
class SourceOrigin;

}

namespace WebCore {

class JSDOMGlobalObject;
class ScriptExecutionContext;

class ScriptModuleLoader final : private ModuleScriptLoaderClient {
    WTF_MAKE_TZONE_ALLOCATED(ScriptModuleLoader);
    WTF_MAKE_NONCOPYABLE(ScriptModuleLoader);
public:
    enum class OwnerType : uint8_t { Document, WorkerOrWorklet };
    enum class ModuleType : uint8_t { Invalid, JavaScript, WebAssembly, JSON };
    explicit ScriptModuleLoader(ScriptExecutionContext*, OwnerType);
    ~ScriptModuleLoader();

    UniqueRef<ScriptModuleLoader> shadowRealmLoader(JSC::JSGlobalObject* realmGlobal) const;

    ScriptExecutionContext* context() { return m_context.get(); }

    JSC::Identifier resolve(JSC::JSGlobalObject*, JSC::JSModuleLoader*, JSC::JSValue moduleName, JSC::JSValue importerModuleKey, JSC::JSValue scriptFetcher);
    JSC::JSInternalPromise* fetch(JSC::JSGlobalObject*, JSC::JSModuleLoader*, JSC::JSValue moduleKey, JSC::JSValue parameters, JSC::JSValue scriptFetcher);
    JSC::JSValue evaluate(JSC::JSGlobalObject*, JSC::JSModuleLoader*, JSC::JSValue moduleKey, JSC::JSValue moduleRecord, JSC::JSValue scriptFetcher, JSC::JSValue awaitedValue, JSC::JSValue resumeMode);
    JSC::JSInternalPromise* importModule(JSC::JSGlobalObject*, JSC::JSModuleLoader*, JSC::JSString*, JSC::JSValue parameters, const JSC::SourceOrigin&);
    JSC::JSObject* createImportMetaProperties(JSC::JSGlobalObject*, JSC::JSModuleLoader*, JSC::JSValue, JSC::JSModuleRecord*, JSC::JSValue);

private:
    void notifyFinished(ModuleScriptLoader&, URL&&, Ref<DeferredPromise>) final;

    URL moduleURL(JSC::JSGlobalObject&, JSC::JSValue);
    URL responseURLFromRequestURL(JSC::JSGlobalObject&, JSC::JSValue);

    WeakPtr<ScriptExecutionContext> m_context;
    MemoryCompactRobinHoodHashMap<String, URL> m_requestURLToResponseURLMap;
    UncheckedKeyHashSet<Ref<ModuleScriptLoader>> m_loaders;
    OwnerType m_ownerType;
    JSC::JSGlobalObject* m_shadowRealmGlobal { nullptr };
};

} // namespace WebCore
