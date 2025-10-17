/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 22, 2025.
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
#include "config.h"
#include "LoadableScript.h"

#include "CachedResourceLoader.h"
#include "CachedScript.h"
#include "ContentSecurityPolicy.h"
#include "Document.h"
#include "LoadableScriptClient.h"
#include "Settings.h"

namespace WebCore {

LoadableScript::LoadableScript(const AtomString& nonce, ReferrerPolicy policy, RequestPriority fetchPriority, const AtomString& crossOriginMode, const AtomString& charset, const AtomString& initiatorType, bool isInUserAgentShadowTree)
    : ScriptElementCachedScriptFetcher(nonce, policy, fetchPriority, crossOriginMode, charset, initiatorType, isInUserAgentShadowTree)
{
}

LoadableScript::~LoadableScript() = default;

void LoadableScript::addClient(LoadableScriptClient& client)
{
    m_clients.add(client);
    if (isLoaded()) {
        Ref protectedThis { *this };
        client.notifyFinished(*this);
    }
}

void LoadableScript::removeClient(LoadableScriptClient& client)
{
    m_clients.remove(client);
}

void LoadableScript::notifyClientFinished()
{
    Ref protectedThis { *this };
    auto clients = WTF::map(m_clients, [](auto& entry) -> WeakPtr<LoadableScriptClient> {
        return entry.key;
    });
    for (auto& client : clients) {
        if (client)
            client->notifyFinished(*this);
    }
}

}
