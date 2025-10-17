/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 3, 2023.
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
#include "CachedScript.h"
#include "Document.h"
#include "LoadableScript.h"
#include "LoadableScriptError.h"
#include "ReferrerPolicy.h"
#include <wtf/TypeCasts.h>

namespace WebCore {

class WeakPtrImplWithEventTargetData;

// A CachedResourceHandle alone does not prevent the underlying CachedResource
// from purging its data buffer. This class holds a client until this class is
// destroyed in order to guarantee that the data buffer will not be purged.
class LoadableNonModuleScriptBase : public LoadableScript, protected CachedResourceClient {
public:
    virtual ~LoadableNonModuleScriptBase();

    bool isLoaded() const final;
    bool hasError() const final;
    std::optional<Error> takeError() final;
    bool wasCanceled() const final;

    Document* document() { return m_weakDocument.get(); }
    CachedScript& cachedScript() { return *m_cachedScript; }
    CachedResourceHandle<CachedScript> protectedCachedScript() { return cachedScript(); }

    bool load(Document&, const URL&);
    bool isAsync() const { return m_isAsync; }
    const AtomString& integrity() const { return m_integrity; }

protected:
    LoadableNonModuleScriptBase(const AtomString& nonce, const AtomString& integrity, ReferrerPolicy, RequestPriority, const AtomString& crossOriginMode, const AtomString& charset, const AtomString& initiatorType, bool isInUserAgentShadowTree, bool isAsync);

private:
    void notifyFinished(CachedResource&, const NetworkLoadMetrics&, LoadWillContinueInAnotherProcess) final;

protected:
    CachedResourceHandle<CachedScript> m_cachedScript { };
    WeakPtr<Document, WeakPtrImplWithEventTargetData> m_weakDocument;
    std::optional<Error> m_error { std::nullopt };
    const AtomString m_integrity;
    bool m_isAsync { false };
};


class LoadableClassicScript final : public LoadableNonModuleScriptBase {
public:
    static Ref<LoadableClassicScript> create(const AtomString& nonce, const AtomString& integrity, ReferrerPolicy, RequestPriority, const AtomString& crossOriginMode, const AtomString& charset, const AtomString& initiatorType, bool isInUserAgentShadowTree, bool isAsync);

    ScriptType scriptType() const final { return ScriptType::Classic; }

    void execute(ScriptElement&) final;

private:
    LoadableClassicScript(const AtomString& nonce, const AtomString& integrity, ReferrerPolicy, RequestPriority, const AtomString& crossOriginMode, const AtomString& charset, const AtomString& initiatorType, bool isInUserAgentShadowTree, bool isAsync);
};

}

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::LoadableClassicScript)
    static bool isType(const WebCore::LoadableScript& script) { return script.isClassicScript(); }
SPECIALIZE_TYPE_TRAITS_END()
