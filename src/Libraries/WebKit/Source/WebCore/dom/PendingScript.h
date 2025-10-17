/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 23, 2024.
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
#include "LoadableScriptClient.h"
#include <wtf/CheckedPtr.h>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/text/TextPosition.h>

namespace WebCore {

class CachedScript;
class PendingScriptClient;
class ScriptElement;

// A container for scripts which may be loaded and executed.
// This can hold LoadableScript and non external inline script.
class PendingScript final : public RefCounted<PendingScript>, public LoadableScriptClient {
public:
    static Ref<PendingScript> create(ScriptElement&, LoadableScript&);
    static Ref<PendingScript> create(ScriptElement&, TextPosition scriptStartPosition);

    virtual ~PendingScript();

    TextPosition startingPosition() const { return m_startingPosition; }
    void setStartingPosition(const TextPosition& position) { m_startingPosition = position; }

    bool watchingForLoad() const { return needsLoading() && m_client; }

    ScriptElement& element() { return m_element.get(); }
    const ScriptElement& element() const { return m_element.get(); }
    Ref<ScriptElement> protectedElement() { return m_element; }
    Ref<const ScriptElement> protectedElement() const { return m_element; }

    LoadableScript* loadableScript() const;
    bool needsLoading() const { return loadableScript(); }

    bool isLoaded() const;
    bool hasError() const;

    void notifyFinished(LoadableScript&) override;

    void setClient(PendingScriptClient&);
    void clearClient();

private:
    PendingScript(ScriptElement&, LoadableScript&);
    PendingScript(ScriptElement&, TextPosition startingPosition);

    void notifyClientFinished();

    Ref<ScriptElement> m_element;
    TextPosition m_startingPosition; // Only used for inline script tags.
    RefPtr<LoadableScript> m_loadableScript;
    CheckedPtr<PendingScriptClient> m_client;
};

inline LoadableScript* PendingScript::loadableScript() const
{
    return m_loadableScript.get();
}

}
