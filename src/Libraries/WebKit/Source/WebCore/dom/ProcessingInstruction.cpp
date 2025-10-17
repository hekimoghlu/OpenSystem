/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 10, 2024.
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
#include "ProcessingInstruction.h"

#include "CSSStyleSheet.h"
#include "CachedCSSStyleSheet.h"
#include "CachedResourceLoader.h"
#include "CachedResourceRequest.h"
#include "CachedXSLStyleSheet.h"
#include "CommonAtomStrings.h"
#include "DocumentInlines.h"
#include "FrameLoader.h"
#include "LocalFrame.h"
#include "MediaQueryParser.h"
#include "MediaQueryParserContext.h"
#include "StyleScope.h"
#include "StyleSheetContents.h"
#include "XMLDocumentParser.h"
#include "XSLStyleSheet.h"
#include <wtf/SetForScope.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(ProcessingInstruction);

inline ProcessingInstruction::ProcessingInstruction(Document& document, String&& target, String&& data)
    : CharacterData(document, WTFMove(data), PROCESSING_INSTRUCTION_NODE)
    , m_target(WTFMove(target))
{
}

Ref<ProcessingInstruction> ProcessingInstruction::create(Document& document, String&& target, String&& data)
{
    return adoptRef(*new ProcessingInstruction(document, WTFMove(target), WTFMove(data)));
}

ProcessingInstruction::~ProcessingInstruction()
{
    if (RefPtr sheet = m_sheet)
        sheet->clearOwnerNode();

    if (CachedResourceHandle cachedSheet = m_cachedSheet)
        cachedSheet->removeClient(*this);

    if (isConnected())
        document().checkedStyleScope()->removeStyleSheetCandidateNode(*this);
}

String ProcessingInstruction::nodeName() const
{
    return m_target;
}

Ref<Node> ProcessingInstruction::cloneNodeInternal(TreeScope& treeScope, CloningOperation)
{
    // FIXME: Is it a problem that this does not copy m_localHref?
    // What about other data members?
    return create(treeScope.documentScope(), String { m_target }, String { data() });
}

void ProcessingInstruction::checkStyleSheet()
{
    Ref document = this->document();
    if (m_target == "xml-stylesheet"_s && document->frame() && parentNode() == document.ptr()) {
        // see http://www.w3.org/TR/xml-stylesheet/
        // ### support stylesheet included in a fragment of this (or another) document
        // ### make sure this gets called when adding from javascript
        auto attributes = parseAttributes(document->cachedResourceLoader(), data());
        if (!attributes)
            return;
        String type = attributes->get<HashTranslatorASCIILiteral>("type"_s);

        m_isCSS = type.isEmpty() || type == cssContentTypeAtom();
#if ENABLE(XSLT)
        m_isXSL = type == "text/xml"_s || type == "text/xsl"_s || type == "application/xml"_s || type == "application/xhtml+xml"_s || type == "application/rss+xml"_s || type == "application/atom+xml"_s;
        if (!m_isCSS && !m_isXSL)
#else
        if (!m_isCSS)
#endif
            return;

        String href = attributes->get<HashTranslatorASCIILiteral>("href"_s);
        String alternate = attributes->get<HashTranslatorASCIILiteral>("alternate"_s);
        m_alternate = alternate == "yes"_s;
        m_title = attributes->get<HashTranslatorASCIILiteral>("title"_s);
        m_media = attributes->get<HashTranslatorASCIILiteral>("media"_s);

        if (m_alternate && m_title.isEmpty())
            return;

        if (href.length() > 1 && href[0] == '#') {
            m_localHref = href.substring(1);
#if ENABLE(XSLT)
            // We need to make a synthetic XSLStyleSheet that is embedded.  It needs to be able
            // to kick off import/include loads that can hang off some parent sheet.
            if (m_isXSL) {
                URL finalURL({ }, m_localHref);
                m_sheet = XSLStyleSheet::createEmbedded(*this, finalURL);
                m_loading = false;
                document->scheduleToApplyXSLTransforms();
            }
#endif
        } else {
            if (CachedResourceHandle cachedSheet = std::exchange(m_cachedSheet, nullptr))
                cachedSheet->removeClient(*this);
            
            if (!m_loading) {
                m_loading = true;
                document->checkedStyleScope()->addPendingSheet(*this);
            }

            ASSERT_WITH_SECURITY_IMPLICATION(!m_cachedSheet);

#if ENABLE(XSLT)
            if (m_isXSL) {
                auto options = CachedResourceLoader::defaultCachedResourceOptions();
                options.mode = FetchOptions::Mode::SameOrigin;
                m_cachedSheet = document->protectedCachedResourceLoader()->requestXSLStyleSheet({ ResourceRequest(document->completeURL(href)), options }).value_or(nullptr);
            } else
#endif
            {
                String charset = attributes->get<HashTranslatorASCIILiteral>("charset"_s);
                CachedResourceRequest request(document->completeURL(href), CachedResourceLoader::defaultCachedResourceOptions(), std::nullopt, charset.isEmpty() ? String::fromLatin1(document->charset()) : WTFMove(charset));

                m_cachedSheet = document->protectedCachedResourceLoader()->requestCSSStyleSheet(WTFMove(request)).value_or(nullptr);
            }
            if (CachedResourceHandle cachedSheet = m_cachedSheet)
                cachedSheet->addClient(*this);
            else {
                // The request may have been denied if (for example) the stylesheet is local and the document is remote.
                m_loading = false;
                document->checkedStyleScope()->removePendingSheet(*this);
#if ENABLE(XSLT)
                if (m_isXSL)
                    document->scheduleToApplyXSLTransforms();
#endif
            }
        }
    }
}

bool ProcessingInstruction::isLoading() const
{
    if (m_loading)
        return true;
    return m_sheet && m_sheet->isLoading();
}

bool ProcessingInstruction::sheetLoaded()
{
    if (!isLoading()) {
        Ref document = this->document();
        if (CheckedRef styleScope = document->styleScope(); styleScope->hasPendingSheet(*this))
            styleScope->removePendingSheet(*this);
#if ENABLE(XSLT)
        if (m_isXSL)
            document->scheduleToApplyXSLTransforms();
#endif
        return true;
    }
    return false;
}

void ProcessingInstruction::setCSSStyleSheet(const String& href, const URL& baseURL, ASCIILiteral charset, const CachedCSSStyleSheet* sheet)
{
    if (!isConnected()) {
        ASSERT(!m_sheet);
        return;
    }

    ASSERT(sheet);

    Ref document = this->document();
    ASSERT(m_isCSS);
    CSSParserContext parserContext(document, baseURL, charset);

    Ref cssSheet = CSSStyleSheet::create(StyleSheetContents::create(href, parserContext), *this, sheet->isCORSSameOrigin());
    cssSheet->setDisabled(m_alternate);
    cssSheet->setTitle(m_title);
    cssSheet->setMediaQueries(MQ::MediaQueryParser::parse(m_media, MediaQueryParserContext(document)));

    m_sheet = WTFMove(cssSheet);

    // We don't need the cross-origin security check here because we are
    // getting the sheet text in "strict" mode. This enforces a valid CSS MIME
    // type.
    parseStyleSheet(sheet->sheetText());
}

#if ENABLE(XSLT)
void ProcessingInstruction::setXSLStyleSheet(const String& href, const URL& baseURL, const String& sheet)
{
    ASSERT(m_isXSL);
    m_sheet = XSLStyleSheet::create(*this, href, baseURL);
    Ref protectedDocument { document() };
    parseStyleSheet(sheet);
}
#endif

RefPtr<StyleSheet> ProcessingInstruction::protectedSheet() const
{
    return m_sheet;
}

void ProcessingInstruction::parseStyleSheet(const String& sheet)
{
    Ref styleSheet = *m_sheet;
    if (m_isCSS)
        downcast<CSSStyleSheet>(styleSheet.get()).protectedContents()->parseString(sheet);
#if ENABLE(XSLT)
    else if (m_isXSL)
        downcast<XSLStyleSheet>(styleSheet.get()).parseString(sheet);
#endif

    if (CachedResourceHandle cachedSheet = std::exchange(m_cachedSheet, nullptr))
        cachedSheet->removeClient(*this);

    m_loading = false;

    if (m_isCSS)
        downcast<CSSStyleSheet>(styleSheet.get()).protectedContents()->checkLoaded();
#if ENABLE(XSLT)
    else if (m_isXSL)
        downcast<XSLStyleSheet>(styleSheet.get()).checkLoaded();
#endif
}

void ProcessingInstruction::addSubresourceAttributeURLs(ListHashSet<URL>& urls) const
{
    if (!sheet())
        return;
    
    addSubresourceURL(urls, sheet()->baseURL());
}

Node::InsertedIntoAncestorResult ProcessingInstruction::insertedIntoAncestor(InsertionType insertionType, ContainerNode& parentOfInsertedTree)
{
    CharacterData::insertedIntoAncestor(insertionType, parentOfInsertedTree);
    if (!insertionType.connectedToDocument)
        return InsertedIntoAncestorResult::Done;
    protectedDocument()->checkedStyleScope()->addStyleSheetCandidateNode(*this, m_createdByParser);
    return InsertedIntoAncestorResult::NeedsPostInsertionCallback;
}

void ProcessingInstruction::didFinishInsertingNode()
{
    checkStyleSheet();
}

void ProcessingInstruction::removedFromAncestor(RemovalType removalType, ContainerNode& oldParentOfRemovedTree)
{
    CharacterData::removedFromAncestor(removalType, oldParentOfRemovedTree);
    if (!removalType.disconnectedFromDocument)
        return;
    
    CheckedRef styleScope = document().styleScope();
    styleScope->removeStyleSheetCandidateNode(*this);

    if (RefPtr sheet = std::exchange(m_sheet, nullptr)) {
        ASSERT(sheet->ownerNode() == this);
        sheet->clearOwnerNode();
    }

    if (m_loading) {
        m_loading = false;
        styleScope->removePendingSheet(*this);
    }

    styleScope->didChangeActiveStyleSheetCandidates();
}

} // namespace
