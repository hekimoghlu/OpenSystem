/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 26, 2022.
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

#include "CachedResourceClient.h"
#include "CachedResourceHandle.h"
#include "CachedScriptFetcher.h"
#include "ModuleScriptLoader.h"
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>
#include <wtf/URL.h>

namespace JSC {
class ScriptFetchParameters;
}
namespace WebCore {

class ModuleScriptLoaderClient;
class CachedScript;
class DeferredPromise;
class Document;
class JSDOMGlobalObject;

class CachedModuleScriptLoader final : public ModuleScriptLoader, private CachedResourceClient {
public:
    static Ref<CachedModuleScriptLoader> create(ModuleScriptLoaderClient&, DeferredPromise&, CachedScriptFetcher&, RefPtr<JSC::ScriptFetchParameters>&&);

    virtual ~CachedModuleScriptLoader();

    bool load(Document&, URL&& sourceURL, std::optional<ServiceWorkersMode>);

    CachedScript* cachedScript() { return m_cachedScript.get(); }
    CachedScriptFetcher& scriptFetcher() { return static_cast<CachedScriptFetcher&>(ModuleScriptLoader::scriptFetcher()); }

private:
    CachedModuleScriptLoader(ModuleScriptLoaderClient&, DeferredPromise&, CachedScriptFetcher&, RefPtr<JSC::ScriptFetchParameters>&&);

    void notifyFinished(CachedResource&, const NetworkLoadMetrics&, LoadWillContinueInAnotherProcess) final;

    CachedResourceHandle<CachedScript> m_cachedScript;
    URL m_sourceURL;
};

} // namespace WebCore
