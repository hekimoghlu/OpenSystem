/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 19, 2022.
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
#include "ResourceLoaderIdentifier.h"
#include "ScriptBuffer.h"
#include "ScriptExecutionContextIdentifier.h"
#include "WorkerScriptFetcher.h"
#include "WorkerScriptLoaderClient.h"
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>
#include <wtf/URL.h>

namespace WebCore {

class DeferredPromise;
class JSDOMGlobalObject;
class ModuleFetchParameters;
class ScriptExecutionContext;
class ModuleScriptLoaderClient;
class WorkerScriptLoader;

class WorkerModuleScriptLoader final : public ModuleScriptLoader, private WorkerScriptLoaderClient {
public:
    static Ref<WorkerModuleScriptLoader> create(ModuleScriptLoaderClient&, DeferredPromise&, WorkerScriptFetcher&, RefPtr<JSC::ScriptFetchParameters>&&);

    virtual ~WorkerModuleScriptLoader();

    void load(ScriptExecutionContext&, URL&& sourceURL);

    WorkerScriptLoader& scriptLoader() { return m_scriptLoader.get(); }
    Ref<WorkerScriptLoader> protectedScriptLoader();

    static String taskMode();
    ReferrerPolicy referrerPolicy();
    bool failed() const { return m_failed; }
    bool retrievedFromServiceWorkerCache() const { return m_retrievedFromServiceWorkerCache; }

    const ScriptBuffer& script() { return m_script; }
    const URL& responseURL() const { return m_responseURL; }
    const String& responseMIMEType() const { return m_responseMIMEType; }

private:
    WorkerModuleScriptLoader(ModuleScriptLoaderClient&, DeferredPromise&, WorkerScriptFetcher&, RefPtr<JSC::ScriptFetchParameters>&&);

    void didReceiveResponse(ScriptExecutionContextIdentifier, std::optional<ResourceLoaderIdentifier>, const ResourceResponse&) final { }
    void notifyFinished(std::optional<ScriptExecutionContextIdentifier>) final;

    void notifyClientFinished();

    Ref<WorkerScriptLoader> m_scriptLoader;
    URL m_sourceURL;
    ScriptBuffer m_script;
    URL m_responseURL;
    String m_responseMIMEType;
    bool m_failed { false };
    bool m_retrievedFromServiceWorkerCache { false };
};

} // namespace WebCore
