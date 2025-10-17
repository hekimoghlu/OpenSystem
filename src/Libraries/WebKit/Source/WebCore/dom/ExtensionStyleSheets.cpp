/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 24, 2025.
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
#include "ExtensionStyleSheets.h"

#include "CSSStyleSheet.h"
#include "Element.h"
#include "HTMLLinkElement.h"
#include "HTMLStyleElement.h"
#include "Page.h"
#include "ProcessingInstruction.h"
#include "SVGStyleElement.h"
#include "Settings.h"
#include "StyleInvalidator.h"
#include "StyleResolver.h"
#include "StyleScope.h"
#include "StyleSheetContents.h"
#include "StyleSheetList.h"
#include "UserContentController.h"
#include "UserContentURLPattern.h"
#include "UserStyleSheet.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ExtensionStyleSheets);

#if ENABLE(CONTENT_EXTENSIONS)
using namespace ContentExtensions;
#endif
using namespace HTMLNames;

ExtensionStyleSheets::ExtensionStyleSheets(Document& document)
    : m_document(document)
{
}

ExtensionStyleSheets::~ExtensionStyleSheets() = default;

Ref<Document> ExtensionStyleSheets::protectedDocument() const
{
    return const_cast<Document&>(m_document.get());
}

static Ref<CSSStyleSheet> createExtensionsStyleSheet(Document& document, URL url, const String& text, UserStyleLevel level)
{
    auto contents = StyleSheetContents::create(url.string(), CSSParserContext(document, url));
    auto styleSheet = CSSStyleSheet::create(contents.get(), document, true);

    contents->setIsUserStyleSheet(level == UserStyleLevel::User);
    contents->parseString(text);

    return styleSheet;
}

CSSStyleSheet* ExtensionStyleSheets::pageUserSheet()
{
    if (m_pageUserSheet)
        return m_pageUserSheet.get();
    
    RefPtr owningPage = m_document->page();
    if (!owningPage)
        return nullptr;
    
    String userSheetText = owningPage->userStyleSheet();
    if (userSheetText.isEmpty())
        return nullptr;
    
    m_pageUserSheet = createExtensionsStyleSheet(protectedDocument().get(), m_document->settings().userStyleSheetLocation(), userSheetText, UserStyleLevel::User);

    return m_pageUserSheet.get();
}

void ExtensionStyleSheets::clearPageUserSheet()
{
    if (m_pageUserSheet) {
        m_pageUserSheet = nullptr;
        protectedDocument()->checkedStyleScope()->didChangeStyleSheetEnvironment();
    }
}

void ExtensionStyleSheets::updatePageUserSheet()
{
    clearPageUserSheet();
    if (pageUserSheet())
        protectedDocument()->checkedStyleScope()->didChangeStyleSheetEnvironment();
}

const Vector<RefPtr<CSSStyleSheet>>& ExtensionStyleSheets::injectedUserStyleSheets() const
{
    updateInjectedStyleSheetCache();
    return m_injectedUserStyleSheets;
}

const Vector<RefPtr<CSSStyleSheet>>& ExtensionStyleSheets::injectedAuthorStyleSheets() const
{
    updateInjectedStyleSheetCache();
    return m_injectedAuthorStyleSheets;
}

void ExtensionStyleSheets::updateInjectedStyleSheetCache() const
{
    if (m_injectedStyleSheetCacheValid)
        return;

    m_injectedStyleSheetCacheValid = true;
    m_injectedUserStyleSheets.clear();
    m_injectedAuthorStyleSheets.clear();
    m_injectedStyleSheetToSource.clear();

    RefPtr owningPage = m_document->page();
    if (!owningPage)
        return;

    auto addStyleSheet = [&](const UserStyleSheet& userStyleSheet) {
        Ref sheet = createExtensionsStyleSheet(protectedDocument().get(), userStyleSheet.url(), userStyleSheet.source(), userStyleSheet.level());

        m_injectedStyleSheetToSource.set(sheet.copyRef(), userStyleSheet.source());

        if (sheet->contents().isUserStyleSheet())
            m_injectedUserStyleSheets.append(WTFMove(sheet));
        else
            m_injectedAuthorStyleSheets.append(WTFMove(sheet));
    };

    for (const auto& userStyleSheet : m_pageSpecificStyleSheets)
        addStyleSheet(userStyleSheet);

    owningPage->protectedUserContentProvider()->forEachUserStyleSheet([&](const UserStyleSheet& userStyleSheet) {
        if (userStyleSheet.pageID())
            return;

        if (userStyleSheet.injectedFrames() == UserContentInjectedFrames::InjectInTopFrameOnly && m_document->ownerElement())
            return;

        auto url = m_document->url();

        if (RefPtr parentDocument = m_document->parentDocument()) {
            switch (userStyleSheet.matchParentFrame()) {
            case UserContentMatchParentFrame::ForOpaqueOrigins:
                if (url.protocolIsAbout() || url.protocolIsBlob() || url.protocolIsData())
                    url = parentDocument->url();
                break;

            case UserContentMatchParentFrame::ForAboutBlank:
                if (url.isAboutBlank())
                    url = parentDocument->url();
                break;

            case UserContentMatchParentFrame::Never:
                break;
            }
        }

        if (!UserContentURLPattern::matchesPatterns(url, userStyleSheet.allowlist(), userStyleSheet.blocklist()))
            return;

        addStyleSheet(userStyleSheet);
    });
}

void ExtensionStyleSheets::injectPageSpecificUserStyleSheet(const UserStyleSheet& userStyleSheet)
{
    m_pageSpecificStyleSheets.append(userStyleSheet);
    invalidateInjectedStyleSheetCache();
}

void ExtensionStyleSheets::removePageSpecificUserStyleSheet(const UserStyleSheet& userStyleSheet)
{
    bool removedStyleSheet = m_pageSpecificStyleSheets.removeFirstMatching([&](const auto& styleSheet) {
        return styleSheet.url() == userStyleSheet.url();
    });

    if (removedStyleSheet)
        invalidateInjectedStyleSheetCache();
}

void ExtensionStyleSheets::invalidateInjectedStyleSheetCache()
{
    m_injectedStyleSheetCacheValid = false;
    protectedDocument()->checkedStyleScope()->didChangeStyleSheetEnvironment();
}

void ExtensionStyleSheets::addUserStyleSheet(Ref<StyleSheetContents>&& userSheet)
{
    ASSERT(userSheet.get().isUserStyleSheet());
    m_userStyleSheets.append(CSSStyleSheet::create(WTFMove(userSheet), protectedDocument().get()));
    protectedDocument()->checkedStyleScope()->didChangeStyleSheetEnvironment();
}

void ExtensionStyleSheets::addAuthorStyleSheetForTesting(Ref<StyleSheetContents>&& authorSheet)
{
    ASSERT(!authorSheet.get().isUserStyleSheet());
    m_authorStyleSheetsForTesting.append(CSSStyleSheet::create(WTFMove(authorSheet), protectedDocument().get()));
    protectedDocument()->checkedStyleScope()->didChangeStyleSheetEnvironment();
}

#if ENABLE(CONTENT_EXTENSIONS)
void ExtensionStyleSheets::addDisplayNoneSelector(const String& identifier, const String& selector, uint32_t selectorID)
{
    auto result = m_contentExtensionSelectorSheets.add(identifier, nullptr);
    if (result.isNewEntry) {
        result.iterator->value = ContentExtensionStyleSheet::create(protectedDocument().get());
        m_userStyleSheets.append(&result.iterator->value->styleSheet());
    }

    if (result.iterator->value->addDisplayNoneSelector(selector, selectorID))
        protectedDocument()->checkedStyleScope()->didChangeStyleSheetEnvironment();
}

void ExtensionStyleSheets::maybeAddContentExtensionSheet(const String& identifier, StyleSheetContents& sheet)
{
    ASSERT(sheet.isUserStyleSheet());

    if (m_contentExtensionSheets.contains(identifier))
        return;

    Ref<CSSStyleSheet> cssSheet = CSSStyleSheet::create(sheet, protectedDocument().get());
    m_contentExtensionSheets.set(identifier, &cssSheet.get());
    m_userStyleSheets.append(adoptRef(cssSheet.leakRef()));
    protectedDocument()->checkedStyleScope()->didChangeStyleSheetEnvironment();

}
#endif // ENABLE(CONTENT_EXTENSIONS)

String ExtensionStyleSheets::contentForInjectedStyleSheet(const RefPtr<CSSStyleSheet>& styleSheet) const
{
    return m_injectedStyleSheetToSource.get(styleSheet);
}

void ExtensionStyleSheets::detachFromDocument()
{
    if (m_pageUserSheet)
        m_pageUserSheet->detachFromDocument();
    for (auto& sheet : m_injectedUserStyleSheets)
        sheet->detachFromDocument();
    for (auto& sheet :  m_injectedAuthorStyleSheets)
        sheet->detachFromDocument();
    for (auto& sheet : m_userStyleSheets)
        sheet->detachFromDocument();
    for (auto& sheet : m_authorStyleSheetsForTesting)
        sheet->detachFromDocument();
}

}
