/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 10, 2023.
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

#include "FetchOptions.h"
#include "LoadableScript.h"
#include "LoadableScriptError.h"
#include "ModuleFetchParameters.h"
#include <JavaScriptCore/ScriptFetcher.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class CachedScript;
class Document;

class WorkerScriptFetcher final : public JSC::ScriptFetcher {
public:
    static Ref<WorkerScriptFetcher> create(Ref<ModuleFetchParameters>&& parameters, FetchOptions::Credentials credentials, FetchOptions::Destination destination, ReferrerPolicy referrerPolicy)
    {
        return adoptRef(*new WorkerScriptFetcher(WTFMove(parameters), credentials, destination, referrerPolicy));
    }

    FetchOptions::Credentials credentials() const { return m_credentials; }
    FetchOptions::Destination destination() const { return m_destination; }
    ReferrerPolicy referrerPolicy() const { return m_referrerPolicy; }

    void notifyLoadCompleted(UniquedStringImpl& moduleKey)
    {
        m_moduleKey = &moduleKey;
        m_isLoaded = true;
    }

    void notifyLoadFailed(LoadableScript::Error&& error)
    {
        m_error = WTFMove(error);
        m_isLoaded = true;
    }

    void notifyLoadWasCanceled()
    {
        m_wasCanceled = true;
        m_isLoaded = true;
    }

    bool isLoaded() const { return m_isLoaded; }
    std::optional<LoadableScript::Error> error() const { return m_error; }
    bool wasCanceled() const { return m_wasCanceled; }
    UniquedStringImpl* moduleKey() const { return m_moduleKey.get(); }
    ModuleFetchParameters& parameters() { return m_parameters.get(); }

    void setReferrerPolicy(ReferrerPolicy referrerPolicy)
    {
        m_referrerPolicy = referrerPolicy;
    }

protected:
    WorkerScriptFetcher(Ref<ModuleFetchParameters>&& parameters, FetchOptions::Credentials credentials, FetchOptions::Destination destination, ReferrerPolicy referrerPolicy)
        : m_credentials(credentials)
        , m_destination(destination)
        , m_referrerPolicy(referrerPolicy)
        , m_parameters(WTFMove(parameters))
    {
    }

private:
    FetchOptions::Credentials m_credentials;
    FetchOptions::Destination m_destination;
    ReferrerPolicy m_referrerPolicy { ReferrerPolicy::EmptyString };
    RefPtr<UniquedStringImpl> m_moduleKey;
    Ref<ModuleFetchParameters> m_parameters;
    std::optional<LoadableScript::Error> m_error;
    bool m_wasCanceled { false };
    bool m_isLoaded { false };
};

} // namespace WebCore
