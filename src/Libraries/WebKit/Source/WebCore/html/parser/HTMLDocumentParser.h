/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 4, 2022.
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

#include "HTMLInputStream.h"
#include "HTMLScriptRunnerHost.h"
#include "HTMLTokenizer.h"
#include "PendingScriptClient.h"
#include "ScriptableDocumentParser.h"
#include <wtf/CheckedRef.h>

namespace WebCore {

class CustomElementRegistry;
class DocumentFragment;
class Element;
class HTMLDocument;
class HTMLParserScheduler;
class HTMLPreloadScanner;
class HTMLScriptRunner;
class HTMLTreeBuilder;
class HTMLResourcePreloader;
class PumpSession;

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(HTMLDocumentParser);
class HTMLDocumentParser : public ScriptableDocumentParser, private HTMLScriptRunnerHost, private PendingScriptClient, public CanMakeCheckedPtr<HTMLDocumentParser> {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(HTMLDocumentParser);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(HTMLDocumentParser);
public:
    static Ref<HTMLDocumentParser> create(HTMLDocument&, OptionSet<ParserContentPolicy> = DefaultParserContentPolicy);
    virtual ~HTMLDocumentParser();

    // CheckedPtr interface
    uint32_t checkedPtrCount() const final { return CanMakeCheckedPtr::checkedPtrCount(); }
    uint32_t checkedPtrCountWithoutThreadCheck() const final { return CanMakeCheckedPtr::checkedPtrCountWithoutThreadCheck(); }
    void incrementCheckedPtrCount() const final { CanMakeCheckedPtr::incrementCheckedPtrCount(); }
    void decrementCheckedPtrCount() const final { CanMakeCheckedPtr::decrementCheckedPtrCount(); }

    HTMLDocumentParser* asHTMLDocumentParser() final { return this; }

    static void parseDocumentFragment(const String&, DocumentFragment&, Element& contextElement, OptionSet<ParserContentPolicy> = { ParserContentPolicy::AllowScriptingContent }, CustomElementRegistry* = nullptr);

    // For HTMLParserScheduler.
    void resumeParsingAfterYield();

    // For HTMLTreeBuilder.
    HTMLTokenizer& tokenizer();
    TextPosition textPosition() const final;

    bool isOnStackOfOpenElements(Element&) const;

protected:
    explicit HTMLDocumentParser(HTMLDocument&, OptionSet<ParserContentPolicy> = DefaultParserContentPolicy);

    void insert(SegmentedString&&) final;
    void append(RefPtr<StringImpl>&&) override;
    void appendSynchronously(RefPtr<StringImpl>&&) override;
    void finish() override;

    HTMLTreeBuilder& treeBuilder();

private:
    HTMLDocumentParser(DocumentFragment&, Element& contextElement, OptionSet<ParserContentPolicy>, CustomElementRegistry*);
    static Ref<HTMLDocumentParser> create(DocumentFragment&, Element& contextElement, OptionSet<ParserContentPolicy>, CustomElementRegistry* = nullptr);

    // DocumentParser
    void detach() final;
    bool hasInsertionPoint() final;
    bool processingData() const final;
    void prepareToStopParsing() final;
    void stopParsing() final;
    bool isWaitingForScripts() const;
    bool isExecutingScript() const final;
    bool hasScriptsWaitingForStylesheets() const final;
    void executeScriptsWaitingForStylesheets() final;
    void suspendScheduledTasks() final;
    void resumeScheduledTasks() final;

    bool shouldAssociateConsoleMessagesWithTextPosition() const final;

    // HTMLScriptRunnerHost
    void watchForLoad(PendingScript&) final;
    void stopWatchingForLoad(PendingScript&) final;
    HTMLInputStream& inputStream() final;
    bool hasPreloadScanner() const final;
    void appendCurrentInputStreamToPreloadScannerAndScan() final;

    // PendingScriptClient
    void notifyFinished(PendingScript&) final;

    Document* contextForParsingSession();

    enum class SynchronousMode : bool { AllowYield, ForceSynchronous };
    void append(RefPtr<StringImpl>&&, SynchronousMode);

    void pumpTokenizer(SynchronousMode);
    bool pumpTokenizerLoop(SynchronousMode, bool parsingFragment, PumpSession&);
    void pumpTokenizerIfPossible(SynchronousMode);
    void constructTreeFromHTMLToken(HTMLTokenizer::TokenPtr&);

    void runScriptsForPausedTreeBuilder();
    void resumeParsingAfterScriptExecution();

    void attemptToEnd();
    void endIfDelayed();
    void attemptToRunDeferredScriptsAndEnd();
    void end();

    bool isParsingFragment() const;
    bool isScheduledForResume() const;
    bool inPumpSession() const;
    bool shouldDelayEnd() const;

    void didBeginYieldingParser() final;
    void didEndYieldingParser() final;

    HTMLParserOptions m_options;
    HTMLInputStream m_input;

    HTMLTokenizer m_tokenizer;
    std::unique_ptr<HTMLScriptRunner> m_scriptRunner;
    std::unique_ptr<HTMLTreeBuilder> m_treeBuilder;
    std::unique_ptr<HTMLPreloadScanner> m_preloadScanner;
    std::unique_ptr<HTMLPreloadScanner> m_insertionPreloadScanner;
    RefPtr<HTMLParserScheduler> m_parserScheduler;
    TextPosition m_textPosition;

    std::unique_ptr<HTMLResourcePreloader> m_preloader;

    bool m_endWasDelayed { false };
    unsigned m_pumpSessionNestingLevel { 0 };
    bool m_shouldEmitTracePoints { false };
};

inline HTMLTokenizer& HTMLDocumentParser::tokenizer()
{
    return m_tokenizer;
}

inline HTMLInputStream& HTMLDocumentParser::inputStream()
{
    return m_input;
}

inline bool HTMLDocumentParser::hasPreloadScanner() const
{
    return m_preloadScanner.get();
}

inline HTMLTreeBuilder& HTMLDocumentParser::treeBuilder()
{
    ASSERT(m_treeBuilder);
    return *m_treeBuilder;
}

} // namespace WebCore
