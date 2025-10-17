/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 21, 2024.
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

#include "PendingScript.h"
#include <wtf/Deque.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/TextPosition.h>

namespace WebCore {

class Document;
class HTMLScriptRunnerHost;
class LocalFrame;
class ScriptSourceCode;
class WeakPtrImplWithEventTargetData;

class HTMLScriptRunner {
    WTF_MAKE_TZONE_ALLOCATED(HTMLScriptRunner);
public:
    HTMLScriptRunner(Document&, HTMLScriptRunnerHost&);
    ~HTMLScriptRunner();

    void detach();

    // Processes the passed in script and any pending scripts if possible.
    void execute(Ref<ScriptElement>&&, const TextPosition& scriptStartPosition);

    void executeScriptsWaitingForLoad(PendingScript&);
    bool hasScriptsWaitingForStylesheets() const { return m_hasScriptsWaitingForStylesheets; }
    void executeScriptsWaitingForStylesheets();
    bool executeScriptsWaitingForParsing();

    bool hasParserBlockingScript() const;
    bool isExecutingScript() const { return !!m_scriptNestingLevel; }

private:
    LocalFrame* frame() const;

    void executePendingScriptAndDispatchEvent(PendingScript&);
    void executeParsingBlockingScripts();

    void requestParsingBlockingScript(ScriptElement&);
    void requestDeferredScript(ScriptElement&);

    void runScript(ScriptElement&, const TextPosition& scriptStartPosition);

    void watchForLoad(PendingScript&);
    void stopWatchingForLoad(PendingScript&);
    bool isPendingScriptReady(const PendingScript&);

    WeakPtr<Document, WeakPtrImplWithEventTargetData> m_document;
    HTMLScriptRunnerHost& m_host;
    RefPtr<PendingScript> m_parserBlockingScript;
    Deque<Ref<PendingScript>> m_scriptsToExecuteAfterParsing; // http://www.whatwg.org/specs/web-apps/current-work/#list-of-scripts-that-will-execute-when-the-document-has-finished-parsing
    unsigned m_scriptNestingLevel;

    // We only want stylesheet loads to trigger script execution if script
    // execution is currently stopped due to stylesheet loads, otherwise we'd
    // cause nested script execution when parsing <style> tags since </style>
    // tags can cause Document to call executeScriptsWaitingForStylesheets.
    bool m_hasScriptsWaitingForStylesheets;
};

} // namespace WebCore
