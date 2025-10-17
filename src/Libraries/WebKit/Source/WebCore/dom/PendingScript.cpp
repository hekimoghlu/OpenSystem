/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 29, 2023.
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
#include "PendingScript.h"

#include "Element.h"
#include "PendingScriptClient.h"
#include "ScriptElement.h"

namespace WebCore {

Ref<PendingScript> PendingScript::create(ScriptElement& element, LoadableScript& loadableScript)
{
    Ref pendingScript = adoptRef(*new PendingScript(element, loadableScript));
    loadableScript.addClient(pendingScript.get());
    return pendingScript;
}

Ref<PendingScript> PendingScript::create(ScriptElement& element, TextPosition scriptStartPosition)
{
    return adoptRef(*new PendingScript(element, scriptStartPosition));
}

PendingScript::PendingScript(ScriptElement& element, TextPosition startingPosition)
    : m_element(element)
    , m_startingPosition(startingPosition)
{
}

PendingScript::PendingScript(ScriptElement& element, LoadableScript& loadableScript)
    : m_element(element)
    , m_loadableScript(&loadableScript)
{
}

PendingScript::~PendingScript()
{
    if (RefPtr loadableScript = m_loadableScript)
        loadableScript->removeClient(*this);
}

void PendingScript::notifyClientFinished()
{
    Ref<PendingScript> protectedThis(*this);
    if (m_client)
        m_client->notifyFinished(*this);
}

void PendingScript::notifyFinished(LoadableScript&)
{
    notifyClientFinished();
}

bool PendingScript::isLoaded() const
{
    return m_loadableScript && m_loadableScript->isLoaded();
}

bool PendingScript::hasError() const
{
    return m_loadableScript && m_loadableScript->hasError();
}

void PendingScript::setClient(PendingScriptClient& client)
{
    ASSERT(!m_client);
    m_client = &client;
    if (isLoaded())
        notifyClientFinished();
}

void PendingScript::clearClient()
{
    ASSERT(m_client);
    m_client = nullptr;
}

}
