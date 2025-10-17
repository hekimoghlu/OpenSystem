/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 5, 2024.
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

#include "JSDOMPromiseDeferred.h"
#include "ModuleFetchParameters.h"
#include <JavaScriptCore/ScriptFetcher.h>
#include <wtf/RefCounted.h>

namespace WebCore {

class ModuleScriptLoaderClient;

class ModuleScriptLoader : public RefCounted<ModuleScriptLoader> {
public:
    virtual ~ModuleScriptLoader() = default;

    void clearClient()
    {
        ASSERT(m_client);
        m_client = nullptr;
    }

    JSC::ScriptFetcher& scriptFetcher() { return m_scriptFetcher.get(); }
    JSC::ScriptFetchParameters* parameters() { return m_parameters.get(); }

protected:
    ModuleScriptLoader(ModuleScriptLoaderClient& client, DeferredPromise& promise, JSC::ScriptFetcher& scriptFetcher, RefPtr<JSC::ScriptFetchParameters>&& parameters)
        : m_client(&client)
        , m_promise(&promise)
        , m_scriptFetcher(scriptFetcher)
        , m_parameters(WTFMove(parameters))
    {
    }

    ModuleScriptLoaderClient* m_client;
    RefPtr<DeferredPromise> m_promise;
    Ref<JSC::ScriptFetcher> m_scriptFetcher;
    RefPtr<JSC::ScriptFetchParameters> m_parameters;
};

} // namespace WebCore
