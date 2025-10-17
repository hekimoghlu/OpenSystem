/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 16, 2025.
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

#include "DecodedDataDocumentParser.h"
#include "ParserContentPolicy.h"
#include "Timer.h"
#include <wtf/text/TextPosition.h>

namespace WebCore {

class ScriptableDocumentParser : public DecodedDataDocumentParser {
public:
    // Only used by Document::open for deciding if its safe to act on a
    // JavaScript document.open() call right now, or it should be ignored.
    virtual bool isExecutingScript() const { return false; }

    virtual TextPosition textPosition() const = 0;

    virtual bool hasScriptsWaitingForStylesheets() const { return false; }

    void executeScriptsWaitingForStylesheetsSoon();

    // Returns true if the parser didn't yield or pause or synchronously execute a script,
    // so calls to PageConsoleClient should be associated with the parser's text position.
    virtual bool shouldAssociateConsoleMessagesWithTextPosition() const = 0;

    void setWasCreatedByScript(bool wasCreatedByScript) { m_wasCreatedByScript = wasCreatedByScript; }
    bool wasCreatedByScript() const { return m_wasCreatedByScript; }

    OptionSet<ParserContentPolicy> parserContentPolicy() const { return m_parserContentPolicy; }
    void setParserContentPolicy(OptionSet<ParserContentPolicy> policy) { m_parserContentPolicy = policy; }

protected:
    explicit ScriptableDocumentParser(Document&, OptionSet<ParserContentPolicy> = { DefaultParserContentPolicy });

    virtual void executeScriptsWaitingForStylesheets() { }

    void detach() override;

private:
    ScriptableDocumentParser* asScriptableDocumentParser() final { return this; }

    void scriptsWaitingForStylesheetsExecutionTimerFired();

    // http://www.whatwg.org/specs/web-apps/current-work/#script-created-parser
    bool m_wasCreatedByScript;
    OptionSet<ParserContentPolicy> m_parserContentPolicy;
    Timer m_scriptsWaitingForStylesheetsExecutionTimer;
};

} // namespace WebCore
