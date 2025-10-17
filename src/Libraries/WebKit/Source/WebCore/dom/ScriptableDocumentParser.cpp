/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 8, 2024.
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
#include "ScriptableDocumentParser.h"

#include "CSSPrimitiveValue.h"
#include "Document.h"
#include "Settings.h"
#include "StyleScope.h"

namespace WebCore {

ScriptableDocumentParser::ScriptableDocumentParser(Document& document, OptionSet<ParserContentPolicy> parserContentPolicy)
    : DecodedDataDocumentParser(document)
    , m_wasCreatedByScript(false)
    , m_parserContentPolicy(parserContentPolicy)
    , m_scriptsWaitingForStylesheetsExecutionTimer(*this, &ScriptableDocumentParser::scriptsWaitingForStylesheetsExecutionTimerFired)
{
    if (scriptingContentIsAllowed(m_parserContentPolicy) && !document.allowsContentJavaScript())
        m_parserContentPolicy = disallowScriptingContent(m_parserContentPolicy);
}

void ScriptableDocumentParser::executeScriptsWaitingForStylesheetsSoon()
{
    ASSERT(!document()->styleScope().hasPendingSheets());

    if (m_scriptsWaitingForStylesheetsExecutionTimer.isActive())
        return;
    if (!hasScriptsWaitingForStylesheets())
        return;

    m_scriptsWaitingForStylesheetsExecutionTimer.startOneShot(0_s);
}

void ScriptableDocumentParser::scriptsWaitingForStylesheetsExecutionTimerFired()
{
    ASSERT(!isDetached());

    RefPtr document = this->document();
    if (!document->styleScope().hasPendingSheets())
        executeScriptsWaitingForStylesheets();

    if (!isDetached())
        document->checkCompleted();
}

void ScriptableDocumentParser::detach()
{
    m_scriptsWaitingForStylesheetsExecutionTimer.stop();

    DecodedDataDocumentParser::detach();
}

};
