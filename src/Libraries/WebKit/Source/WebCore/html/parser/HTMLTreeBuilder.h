/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 28, 2024.
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

#include "HTMLConstructionSite.h"
#include "HTMLParserOptions.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/text/StringBuilder.h>
#include <wtf/text/TextPosition.h>

namespace WebCore {

class JSCustomElementInterface;
class HTMLDocumentParser;
class ScriptElement;

enum class TagName : uint16_t;

struct CustomElementConstructionData {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;

    CustomElementConstructionData(Ref<JSCustomElementInterface>&&, Ref<CustomElementRegistry>&&, const AtomString& name, Vector<Attribute>&&);
    ~CustomElementConstructionData();

    Ref<JSCustomElementInterface> elementInterface;
    Ref<CustomElementRegistry> registry;
    AtomString name;
    Vector<Attribute> attributes;
};

class HTMLTreeBuilder {
    WTF_MAKE_TZONE_ALLOCATED(HTMLTreeBuilder);
public:
    HTMLTreeBuilder(HTMLDocumentParser&, HTMLDocument&, OptionSet<ParserContentPolicy>, const HTMLParserOptions&);
    HTMLTreeBuilder(HTMLDocumentParser&, DocumentFragment&, Element& contextElement, OptionSet<ParserContentPolicy>, const HTMLParserOptions&, CustomElementRegistry*);
    void setShouldSkipLeadingNewline(bool);

    ~HTMLTreeBuilder();

    bool isParsingFragment() const;

    void constructTree(AtomHTMLToken&&);

    bool isParsingTemplateContents() const;
    bool hasParserBlockingScriptWork() const;

    // Must be called to take the parser-blocking script before calling the parser again.
    RefPtr<ScriptElement> takeScriptToProcess(TextPosition& scriptStartPosition);
    const ScriptElement* scriptToProcess() const { return m_scriptToProcess.get(); }

    std::unique_ptr<CustomElementConstructionData> takeCustomElementConstructionData() { return WTFMove(m_customElementToConstruct); }
    void didCreateCustomOrFallbackElement(Ref<Element>&&, CustomElementConstructionData&);

    // Done, close any open tags, etc.
    void finished();

    bool isOnStackOfOpenElements(Element&) const;

private:
    class ExternalCharacterTokenBuffer;

    // Represents HTML5 "insertion mode"
    // http://www.whatwg.org/specs/web-apps/current-work/multipage/parsing.html#insertion-mode
    enum class InsertionMode {
        Initial,
        BeforeHTML,
        BeforeHead,
        InHead,
        InHeadNoscript,
        AfterHead,
        TemplateContents,
        InBody,
        Text,
        InTable,
        InTableText,
        InCaption,
        InColumnGroup,
        InTableBody,
        InRow,
        InCell,
        InSelect,
        InSelectInTable,
        AfterBody,
        InFrameset,
        AfterFrameset,
        AfterAfterBody,
        AfterAfterFrameset,
    };

    bool isParsingFragmentOrTemplateContents() const;

#if ENABLE(TELEPHONE_NUMBER_DETECTION) && PLATFORM(IOS_FAMILY)
    void insertPhoneNumberLink(const String&);
    void linkifyPhoneNumbers(const String&);
#endif

    void processToken(AtomHTMLToken&&);

    void processDoctypeToken(AtomHTMLToken&&);
    void processStartTag(AtomHTMLToken&&);
    void processEndTag(AtomHTMLToken&&);
    void processComment(AtomHTMLToken&&);
    void processCharacter(AtomHTMLToken&&);
    void processEndOfFile(AtomHTMLToken&&);

    bool processStartTagForInHead(AtomHTMLToken&&);
    void processStartTagForInBody(AtomHTMLToken&&);
    void processStartTagForInTable(AtomHTMLToken&&);
    void processEndTagForInBody(AtomHTMLToken&&);
    void processEndTagForInTable(AtomHTMLToken&&);
    void processEndTagForInTableBody(AtomHTMLToken&&);
    void processEndTagForInRow(AtomHTMLToken&&);
    void processEndTagForInCell(AtomHTMLToken&&);

    void processHtmlStartTagForInBody(AtomHTMLToken&&);
    bool processBodyEndTagForInBody(AtomHTMLToken&&);
    bool processTableEndTagForInTable();
    bool processCaptionEndTagForInCaption();
    bool processColgroupEndTagForInColumnGroup();
    bool processTrEndTagForInRow();

    void processAnyOtherEndTagForInBody(AtomHTMLToken&&);

    inline bool consumeAndInsertWhitespace(ExternalCharacterTokenBuffer&);
    void processCharacterBuffer(ExternalCharacterTokenBuffer&);
    inline void processCharacterBufferForInBody(ExternalCharacterTokenBuffer&);

    void processFakeStartTag(TagName, Vector<Attribute>&& attributes = Vector<Attribute>());
    void processFakeEndTag(TagName);
    void processFakeEndTag(const HTMLStackItem&);
    void processFakeCharacters(const String&);
    void processFakePEndTagIfPInButtonScope();

    void processGenericRCDATAStartTag(AtomHTMLToken&&);
    void processGenericRawTextStartTag(AtomHTMLToken&&);
    void processScriptStartTag(AtomHTMLToken&&);

    // Default processing for the different insertion modes.
    void defaultForInitial();
    void defaultForBeforeHTML();
    void defaultForBeforeHead();
    void defaultForInHead();
    void defaultForInHeadNoscript();
    void defaultForAfterHead();
    void defaultForInTableText();

    bool shouldProcessTokenInForeignContent(const AtomHTMLToken&);
    void processTokenInForeignContent(AtomHTMLToken&&);

    HTMLStackItem& adjustedCurrentStackItem();

    void callTheAdoptionAgency(AtomHTMLToken&);

    void closeTheCell();

    template <bool shouldClose(const HTMLStackItem&)> void processCloseWhenNestedTag(AtomHTMLToken&&);

    void parseError(const AtomHTMLToken&);

    void resetInsertionModeAppropriately();

    void insertGenericHTMLElement(AtomHTMLToken&&);

    void processTemplateStartTag(AtomHTMLToken&&);
    bool processTemplateEndTag(AtomHTMLToken&&);
    bool processEndOfFileForInTemplateContents(AtomHTMLToken&&);

    class FragmentParsingContext {
    public:
        FragmentParsingContext();
        FragmentParsingContext(DocumentFragment&, Element& contextElement);

        DocumentFragment* fragment() const;
        Element& contextElement();
        HTMLStackItem& contextElementStackItem();

    private:
        DocumentFragment* m_fragment { nullptr };
        HTMLStackItem m_contextElementStackItem;
    };

    HTMLDocumentParser& m_parser;
    const HTMLParserOptions m_options;
    FragmentParsingContext m_fragmentContext;

    HTMLConstructionSite m_tree;

    // https://html.spec.whatwg.org/multipage/syntax.html#the-insertion-mode
    InsertionMode m_insertionMode { InsertionMode::Initial };
    InsertionMode m_originalInsertionMode { InsertionMode::Initial };
    Vector<InsertionMode, 1> m_templateInsertionModes;

    // https://html.spec.whatwg.org/multipage/syntax.html#concept-pending-table-char-tokens
    StringBuilder m_pendingTableCharacters;

    RefPtr<ScriptElement> m_scriptToProcess; // <script> tag which needs processing before resuming the parser.
    TextPosition m_scriptToProcessStartPosition; // Starting line number of the script tag needing processing.

    std::unique_ptr<CustomElementConstructionData> m_customElementToConstruct;

    bool m_shouldSkipLeadingNewline { false };

    bool m_framesetOk { true };

#if ASSERT_ENABLED
    bool m_destroyed { false };
    bool m_destructionProhibited { true };
#endif
};

inline HTMLTreeBuilder::~HTMLTreeBuilder()
{
#if ASSERT_ENABLED
    ASSERT(!m_destroyed);
    ASSERT(!m_destructionProhibited);
    m_destroyed = true;
#endif
}

inline void HTMLTreeBuilder::setShouldSkipLeadingNewline(bool shouldSkip)
{
    ASSERT(!m_destroyed);
    m_shouldSkipLeadingNewline = shouldSkip;
}

inline bool HTMLTreeBuilder::isParsingFragment() const
{
    ASSERT(!m_destroyed);
    return !!m_fragmentContext.fragment();
}

inline bool HTMLTreeBuilder::hasParserBlockingScriptWork() const
{
    ASSERT(!m_destroyed);
    ASSERT(!(m_scriptToProcess && m_customElementToConstruct));
    return m_scriptToProcess || m_customElementToConstruct;
}

inline DocumentFragment* HTMLTreeBuilder::FragmentParsingContext::fragment() const
{
    return m_fragment;
}

} // namespace WebCore
