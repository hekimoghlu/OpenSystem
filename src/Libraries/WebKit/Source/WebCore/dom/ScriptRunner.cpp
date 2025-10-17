/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 3, 2025.
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
#include "ScriptRunner.h"

#include "Document.h"
#include "Element.h"
#include "PendingScript.h"
#include "ScriptElement.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ScriptRunner);

ScriptRunner::ScriptRunner(Document& document)
    : m_document(document)
    , m_timer(*this, &ScriptRunner::timerFired)
{
}

ScriptRunner::~ScriptRunner()
{
    for (auto& pendingScript : m_scriptsToExecuteSoon) {
        UNUSED_PARAM(pendingScript);
        m_document->decrementLoadEventDelayCount();
    }
    for (auto& pendingScript : m_scriptsToExecuteInOrder) {
        if (pendingScript->watchingForLoad())
            pendingScript->clearClient();
        m_document->decrementLoadEventDelayCount();
    }
    for (auto& pendingScript : m_pendingAsyncScripts) {
        if (pendingScript->watchingForLoad())
            pendingScript->clearClient();
        m_document->decrementLoadEventDelayCount();
    }
}

void ScriptRunner::ref() const
{
    m_document->ref();
}

void ScriptRunner::deref() const
{
    m_document->deref();
}

void ScriptRunner::queueScriptForExecution(ScriptElement& scriptElement, LoadableScript& loadableScript, ExecutionType executionType)
{
    ASSERT(scriptElement.element().isConnected());

    m_document->incrementLoadEventDelayCount();

    Ref pendingScript = PendingScript::create(scriptElement, loadableScript);
    switch (executionType) {
    case ASYNC_EXECUTION:
        m_pendingAsyncScripts.add(pendingScript.copyRef());
        break;
    case IN_ORDER_EXECUTION:
        m_scriptsToExecuteInOrder.append(pendingScript.copyRef());
        break;
    }
    pendingScript->setClient(*this);
}

void ScriptRunner::suspend()
{
    m_timer.stop();
}

void ScriptRunner::resume()
{
    if (hasPendingScripts() && !m_document->hasActiveParserYieldToken())
        m_timer.startOneShot(0_s);
}

void ScriptRunner::documentFinishedParsing()
{
    if (!m_scriptsToExecuteSoon.isEmpty() && !m_timer.isActive())
        resume();
}

void ScriptRunner::notifyFinished(PendingScript& pendingScript)
{
    if (pendingScript.element().willExecuteInOrder())
        ASSERT(!m_scriptsToExecuteInOrder.isEmpty());
    else
        m_scriptsToExecuteSoon.append(m_pendingAsyncScripts.take(pendingScript).releaseNonNull());
    pendingScript.clearClient();

    if (!m_document->hasActiveParserYieldToken())
        m_timer.startOneShot(0_s);
}

void ScriptRunner::timerFired()
{
    Ref document = m_document.get();

    Vector<RefPtr<PendingScript>> scripts;

    if (document->shouldDeferAsynchronousScriptsUntilParsingFinishes()) {
        // Scripts not added by the parser are executed asynchronously and yet do not have the 'async' attribute set.
        // We only want to delay scripts that were explicitly marked as 'async' by the developer.
        m_scriptsToExecuteSoon.removeAllMatching([&](auto& pendingScript) {
            if (pendingScript->element().hasAsyncAttribute())
                return false;
            scripts.append(WTFMove(pendingScript));
            return true;
        });
    } else
        scripts.swap(m_scriptsToExecuteSoon);

    size_t numInOrderScriptsToExecute = 0;
    for (; numInOrderScriptsToExecute < m_scriptsToExecuteInOrder.size() && m_scriptsToExecuteInOrder[numInOrderScriptsToExecute]->isLoaded(); ++numInOrderScriptsToExecute)
        scripts.append(m_scriptsToExecuteInOrder[numInOrderScriptsToExecute].ptr());
    if (numInOrderScriptsToExecute)
        m_scriptsToExecuteInOrder.remove(0, numInOrderScriptsToExecute);

    for (auto& currentScript : scripts) {
        RefPtr script = WTFMove(currentScript);
        ASSERT(script);
        // Paper over https://bugs.webkit.org/show_bug.cgi?id=144050
        if (!script)
            continue;
        ASSERT(script->needsLoading());
        script->protectedElement()->executePendingScript(*script);
        document->decrementLoadEventDelayCount();
    }
}

void ScriptRunner::clearPendingScripts()
{
    m_scriptsToExecuteInOrder.clear();
    m_scriptsToExecuteSoon.clear();
    m_pendingAsyncScripts.clear();
}

} // namespace WebCore
