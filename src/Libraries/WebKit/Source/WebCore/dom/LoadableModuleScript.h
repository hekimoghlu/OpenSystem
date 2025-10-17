/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 6, 2024.
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

#include "LoadableScript.h"
#include "LoadableScriptError.h"
#include <wtf/TypeCasts.h>

namespace WebCore {

class ScriptSourceCode;
class ModuleFetchParameters;

class LoadableModuleScript final : public LoadableScript {
public:
    virtual ~LoadableModuleScript();

    static Ref<LoadableModuleScript> create(const AtomString& nonce, const AtomString& integrity, ReferrerPolicy, RequestPriority, const AtomString& crossOriginMode, const AtomString& charset, const AtomString& initiatorType, bool isInUserAgentShadowTree);

    bool isLoaded() const final;
    bool hasError() const final;
    std::optional<Error> takeError() final;
    bool wasCanceled() const final;

    ScriptType scriptType() const final { return ScriptType::Module; }

    void execute(ScriptElement&) final;

    void notifyLoadCompleted(UniquedStringImpl&);
    void notifyLoadFailed(LoadableScript::Error&&);
    void notifyLoadWasCanceled();

    UniquedStringImpl* moduleKey() const { return m_moduleKey.get(); }

    ModuleFetchParameters& parameters() { return m_parameters.get(); }

private:
    LoadableModuleScript(const AtomString& nonce, const AtomString& integrity, ReferrerPolicy, RequestPriority, const AtomString& crossOriginMode, const AtomString& charset, const AtomString& initiatorType, bool isInUserAgentShadowTree);

    Ref<ModuleFetchParameters> m_parameters;
    RefPtr<UniquedStringImpl> m_moduleKey;
    std::optional<LoadableScript::Error> m_error;
    bool m_wasCanceled { false };
    bool m_isLoaded { false };
};

}

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::LoadableModuleScript)
    static bool isType(const WebCore::LoadableScript& script) { return script.isModuleScript(); }
SPECIALIZE_TYPE_TRAITS_END()
