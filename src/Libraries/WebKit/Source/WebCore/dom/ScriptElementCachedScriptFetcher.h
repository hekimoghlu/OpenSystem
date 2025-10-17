/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 9, 2021.
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

#include "CachedScriptFetcher.h"
#include "ResourceLoaderOptions.h"
#include "ScriptType.h"

namespace WebCore {

class ScriptElementCachedScriptFetcher : public CachedScriptFetcher {
public:
    static const ASCIILiteral defaultCrossOriginModeForModule;

    virtual CachedResourceHandle<CachedScript> requestModuleScript(Document&, const URL& sourceURL, String&& integrity, std::optional<ServiceWorkersMode>) const;

    virtual ScriptType scriptType() const = 0;
    bool isClassicScript() const { return scriptType() == ScriptType::Classic; }
    bool isModuleScript() const { return scriptType() == ScriptType::Module; }
    bool isImportMap() const { return scriptType() == ScriptType::ImportMap; }

    const String& crossOriginMode() const { return m_crossOriginMode; }

protected:
    ScriptElementCachedScriptFetcher(const AtomString& nonce, ReferrerPolicy policy, RequestPriority fetchPriority, const AtomString& crossOriginMode, const AtomString& charset, const AtomString& initiatorType, bool isInUserAgentShadowTree)
        : CachedScriptFetcher(nonce, policy, fetchPriority, charset, initiatorType, isInUserAgentShadowTree)
        , m_crossOriginMode(crossOriginMode)
    {
    }

private:
    const AtomString m_crossOriginMode;
};

} // namespace WebCore
