/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 29, 2024.
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

#include "LocalFrameView.h"
#include "ParserContentPolicy.h"
#include "PendingScriptClient.h"
#include "ScriptableDocumentParser.h"
#include "SegmentedString.h"
#include "XMLErrors.h"
// FIXME (286277): Stop ignoring -Wundef and -Wdeprecated-declarations in code that imports libxml and libxslt headers
IGNORE_WARNINGS_BEGIN("deprecated-declarations")
IGNORE_WARNINGS_BEGIN("undef")
#include <libxml/tree.h>
#include <libxml/xmlstring.h>
IGNORE_WARNINGS_END
IGNORE_WARNINGS_END
#include <wtf/CheckedRef.h>
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/AtomStringHash.h>
#include <wtf/text/CString.h>

namespace WebCore {

class ContainerNode;
class CachedResourceLoader;
class DocumentFragment;
class Element;
class PendingCallbacks;
class Text;

class XMLParserContext : public RefCounted<XMLParserContext> {
public:
    static RefPtr<XMLParserContext> createMemoryParser(xmlSAXHandlerPtr, void* userData, const CString& chunk);
    static Ref<XMLParserContext> createStringParser(xmlSAXHandlerPtr, void* userData);
    XMLParserContext() = delete;
    ~XMLParserContext();
    xmlParserCtxtPtr context() const { return m_context; }

private:
    XMLParserContext(xmlParserCtxtPtr context)
        : m_context(context)
    {
    }
    xmlParserCtxtPtr m_context;
};

class XMLDocumentParser final : public ScriptableDocumentParser, public PendingScriptClient, public CanMakeCheckedPtr<XMLDocumentParser> {
    WTF_MAKE_TZONE_ALLOCATED(XMLDocumentParser);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(XMLDocumentParser);
public:
    enum class IsInFrameView : bool { No, Yes };
    static Ref<XMLDocumentParser> create(Document& document, IsInFrameView isInFrameView, OptionSet<ParserContentPolicy> policy = DefaultParserContentPolicy)
    {
        return adoptRef(*new XMLDocumentParser(document, isInFrameView, policy));
    }
    static Ref<XMLDocumentParser> create(DocumentFragment& fragment, HashMap<AtomString, AtomString>&& prefixToNamespaceMap, const AtomString& defaultNamespaceURI, OptionSet<ParserContentPolicy> parserContentPolicy)
    {
        return adoptRef(*new XMLDocumentParser(fragment, WTFMove(prefixToNamespaceMap), defaultNamespaceURI, parserContentPolicy));
    }

    XMLDocumentParser() = delete;
    ~XMLDocumentParser();

    // Exposed for callbacks:
    void handleError(XMLErrors::Type, const char* message, TextPosition);

    void setIsXHTMLDocument(bool isXHTML) { m_isXHTMLDocument = isXHTML; }
    bool isXHTMLDocument() const { return m_isXHTMLDocument; }

    static bool parseDocumentFragment(const String&, DocumentFragment&, Element* parent = nullptr, OptionSet<ParserContentPolicy> = { ParserContentPolicy::AllowScriptingContent });

    // Used by XMLHttpRequest to check if the responseXML was well formed.
    bool wellFormed() const final { return !m_sawError; }

    static bool supportsXMLVersion(const String&);

private:
    explicit XMLDocumentParser(Document&, IsInFrameView, OptionSet<ParserContentPolicy>);
    XMLDocumentParser(DocumentFragment&, HashMap<AtomString, AtomString>&&, const AtomString&, OptionSet<ParserContentPolicy>);

    // CheckedPtr interface
    uint32_t checkedPtrCount() const final { return CanMakeCheckedPtr::checkedPtrCount(); }
    uint32_t checkedPtrCountWithoutThreadCheck() const final { return CanMakeCheckedPtr::checkedPtrCountWithoutThreadCheck(); }
    void incrementCheckedPtrCount() const final { CanMakeCheckedPtr::incrementCheckedPtrCount(); }
    void decrementCheckedPtrCount() const final { CanMakeCheckedPtr::decrementCheckedPtrCount(); }

    void insert(SegmentedString&&) final;
    void append(RefPtr<StringImpl>&&) final;
    void finish() final;
    void stopParsing() final;
    void detach() final;

    TextPosition textPosition() const final;
    bool shouldAssociateConsoleMessagesWithTextPosition() const final;

    void notifyFinished(PendingScript&) final;

    void end();

    void pauseParsing();
    void resumeParsing();

    bool appendFragmentSource(const String&);

public:
    // Callbacks from parser SAX, and other functions needed inside
    // the parser implementation, but outside this class.

    void error(XMLErrors::Type, const char* message, va_list args) WTF_ATTRIBUTE_PRINTF(3, 0);
    void startElementNs(const xmlChar* xmlLocalName, const xmlChar* xmlPrefix, const xmlChar* xmlURI,
        int numNamespaces, const xmlChar** namespaces,
        int numAttributes, int numDefaulted, const xmlChar** libxmlAttributes);
    void endElementNs();
    void characters(std::span<const xmlChar>);
    void processingInstruction(const xmlChar* target, const xmlChar* data);
    void cdataBlock(const xmlChar*, int length);
    void comment(const xmlChar*);
    void startDocument(const xmlChar* version, const xmlChar* encoding, int standalone);
    void internalSubset(const xmlChar* name, const xmlChar* externalID, const xmlChar* systemID);
    void endDocument();

private:
    void initializeParserContext(const CString& chunk = CString());

    void pushCurrentNode(ContainerNode*);
    void popCurrentNode();
    void clearCurrentNodeStack();

    void insertErrorMessageBlock();

    void createLeafTextNode();
    bool updateLeafTextNode();

    void doWrite(const String&);
    void doEnd();

    xmlParserCtxtPtr context() const { return m_context ? m_context->context() : nullptr; };

    IsInFrameView m_isInFrameView { IsInFrameView::No };

    SegmentedString m_originalSourceForTransform;

    RefPtr<XMLParserContext> m_context;
    std::unique_ptr<PendingCallbacks> m_pendingCallbacks;
    Vector<xmlChar> m_bufferedText;

    CheckedPtr<ContainerNode> m_currentNode;
    Vector<CheckedPtr<ContainerNode>> m_currentNodeStack;

    RefPtr<Text> m_leafTextNode;

    bool m_sawError { false };
    bool m_sawCSS { false };
    bool m_sawXSLTransform { false };
    bool m_sawFirstElement { false };
    bool m_isXHTMLDocument { false };
    bool m_parserPaused { false };
    bool m_requestingScript { false };
    bool m_finishCalled { false };

    std::unique_ptr<XMLErrors> m_xmlErrors;

    RefPtr<PendingScript> m_pendingScript;
    TextPosition m_scriptStartPosition;

    bool m_parsingFragment { false };

    HashMap<AtomString, AtomString> m_prefixToNamespaceMap;
    AtomString m_defaultNamespaceURI;

    SegmentedString m_pendingSrc;
};

#if ENABLE(XSLT)
xmlDocPtr xmlDocPtrForString(CachedResourceLoader&, const String& source, const String& url);
#endif

xmlParserInputPtr externalEntityLoader(const char* url, const char* id, xmlParserCtxtPtr);
void initializeXMLParser();

std::optional<HashMap<String, String>> parseAttributes(CachedResourceLoader&, const String&);

} // namespace WebCore
