/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 6, 2022.
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
#include "InlineStyleSheetOwner.h"

#include "CommonAtomStrings.h"
#include "ContentSecurityPolicy.h"
#include "ElementInlines.h"
#include "Logging.h"
#include "MediaList.h"
#include "MediaQueryParser.h"
#include "MediaQueryParserContext.h"
#include "PluginDocument.h"
#include "ScriptableDocumentParser.h"
#include "ShadowRoot.h"
#include "StyleScope.h"
#include "StyleSheetContents.h"
#include "StyleSheetContentsCache.h"
#include "TextNodeTraversal.h"
#include <wtf/HashMap.h>
#include <wtf/NeverDestroyed.h>

namespace WebCore {

static CSSParserContext parserContextForElement(const Element& element)
{
    auto* shadowRoot = element.containingShadowRoot();
    // User agent shadow trees can't contain document-relative URLs. Use blank URL as base allowing cross-document sharing.
    auto& baseURL = shadowRoot && shadowRoot->mode() == ShadowRootMode::UserAgent ? aboutBlankURL() : element.document().baseURL();

    CSSParserContext result = CSSParserContext { element.document(), baseURL, element.document().characterSetWithUTF8Fallback() };
    if (shadowRoot && shadowRoot->mode() == ShadowRootMode::UserAgent)
        result.mode = UASheetMode;
    return result;
}

InlineStyleSheetOwner::InlineStyleSheetOwner(Document& document, bool createdByParser)
    : m_isParsingChildren(createdByParser)
    , m_loading(false)
    , m_startTextPosition()
{
    if (createdByParser && document.scriptableDocumentParser() && !document.isInDocumentWrite())
        m_startTextPosition = document.scriptableDocumentParser()->textPosition();
}

InlineStyleSheetOwner::~InlineStyleSheetOwner()
{
    if (m_sheet)
        clearSheet();
}

void InlineStyleSheetOwner::insertedIntoDocument(Element& element)
{
    m_styleScope = Style::Scope::forNode(element);
    m_styleScope->addStyleSheetCandidateNode(element, m_isParsingChildren);

    if (m_isParsingChildren)
        return;
    createSheetFromTextContents(element);
}

void InlineStyleSheetOwner::removedFromDocument(Element& element)
{
    if (CheckedPtr scope = m_styleScope.get()) {
        if (scope->hasPendingSheet(element))
            scope->removePendingSheet(element);
        scope->removeStyleSheetCandidateNode(element);
    }
    if (m_sheet)
        clearSheet();
}

void InlineStyleSheetOwner::clearDocumentData(Element& element)
{
    if (RefPtr sheet = m_sheet)
        sheet->clearOwnerNode();

    if (CheckedPtr scope = m_styleScope.get())
        scope->removeStyleSheetCandidateNode(element);
}

void InlineStyleSheetOwner::childrenChanged(Element& element)
{
    if (m_isParsingChildren)
        return;
    if (!element.isConnected())
        return;
    createSheetFromTextContents(element);
}

void InlineStyleSheetOwner::finishParsingChildren(Element& element)
{
    if (element.isConnected())
        createSheetFromTextContents(element);
    m_isParsingChildren = false;
}

void InlineStyleSheetOwner::createSheetFromTextContents(Element& element)
{
    createSheet(element, TextNodeTraversal::contentsAsString(element));
}

void InlineStyleSheetOwner::clearSheet()
{
    ASSERT(m_sheet);
    RefPtr sheet = std::exchange(m_sheet, nullptr);
    sheet->clearOwnerNode();
}

inline bool isValidCSSContentType(const AtomString& type)
{
    // https://html.spec.whatwg.org/multipage/semantics.html#update-a-style-block
    if (type.isEmpty())
        return true;
    return equalLettersIgnoringASCIICase(type, "text/css"_s);
}

void InlineStyleSheetOwner::createSheet(Element& element, const String& text)
{
    ASSERT(element.isConnected());
    Ref document = element.document();
    if (RefPtr sheet = m_sheet) {
        if (sheet->isLoading() && m_styleScope)
            CheckedRef { *m_styleScope }->removePendingSheet(element);
        clearSheet();
    }

    if (!isValidCSSContentType(m_contentType))
        return;

    ASSERT(document->contentSecurityPolicy());
    if (!document->checkedContentSecurityPolicy()->allowInlineStyle(document->url().string(), m_startTextPosition.m_line, text, CheckUnsafeHashes::No, element, element.nonce(), element.isInUserAgentShadowTree() || is<PluginDocument>(document))) {
        element.notifyLoadedSheetAndAllCriticalSubresources(true);
        return;
    }

    auto mediaQueries = MQ::MediaQueryParser::parse(m_media, MediaQueryParserContext(document));

    if (CheckedPtr scope = m_styleScope.get())
        scope->addPendingSheet(element);

    Style::StyleSheetContentsCache::Key cacheKey { text, parserContextForElement(element) };
    if (RefPtr cachedSheet = Style::StyleSheetContentsCache::singleton().get(cacheKey)) {
        ASSERT(cachedSheet->isCacheableWithNoBaseURLDependency());
        Ref sheet = CSSStyleSheet::createInline(*cachedSheet, element, m_startTextPosition);
        m_sheet = sheet.copyRef();
        sheet->setMediaQueries(WTFMove(mediaQueries));
        if (!element.isInShadowTree())
            sheet->setTitle(element.title());

        sheetLoaded(element);
        element.notifyLoadedSheetAndAllCriticalSubresources(false);
        return;
    }

    m_loading = true;

    Ref contents = StyleSheetContents::create(String(), cacheKey.second);

    Ref sheet = CSSStyleSheet::createInline(contents.get(), element, m_startTextPosition);
    m_sheet = sheet.copyRef();
    sheet->setMediaQueries(WTFMove(mediaQueries));
    if (!element.isInShadowTree())
        sheet->setTitle(element.title());

    contents->parseString(text);

    m_loading = false;

    contents->checkLoaded();

    if (contents->isCacheableWithNoBaseURLDependency())
        Style::StyleSheetContentsCache::singleton().add(WTFMove(cacheKey), contents);
}

bool InlineStyleSheetOwner::isLoading() const
{
    if (m_loading)
        return true;
    return m_sheet && m_sheet->isLoading();
}

bool InlineStyleSheetOwner::sheetLoaded(Element& element)
{
    if (isLoading())
        return false;

    if (CheckedPtr scope = m_styleScope.get())
        scope->removePendingSheet(element);

    return true;
}

void InlineStyleSheetOwner::startLoadingDynamicSheet(Element& element)
{
    if (CheckedPtr scope = m_styleScope.get(); scope && !scope->hasPendingSheet(element))
        scope->addPendingSheet(element);
}

}
