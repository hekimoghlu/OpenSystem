/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 3, 2025.
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
#include "UserContentProvider.h"

#include "Chrome.h"
#include "ChromeClient.h"
#include "Document.h"
#include "DocumentLoader.h"
#include "FrameDestructionObserverInlines.h"
#include "FrameLoader.h"
#include "LocalFrame.h"
#include "Page.h"

#if ENABLE(CONTENT_EXTENSIONS)
#include "ContentExtensionCompiler.h"
#include "ContentExtensionsBackend.h"
#include "ContentRuleListResults.h"
#endif

namespace WebCore {

UserContentProvider::UserContentProvider()
{
}

UserContentProvider::~UserContentProvider()
{
    ASSERT(m_pages.isEmptyIgnoringNullReferences());
}

void UserContentProvider::addPage(Page& page)
{
    ASSERT(!m_pages.contains(page));

    m_pages.add(page);
}

void UserContentProvider::removePage(Page& page)
{
    ASSERT(m_pages.contains(page));

    m_pages.remove(page);
}

void UserContentProvider::registerForUserMessageHandlerInvalidation(UserContentProviderInvalidationClient& invalidationClient)
{
    ASSERT(!m_userMessageHandlerInvalidationClients.contains(invalidationClient));

    m_userMessageHandlerInvalidationClients.add(invalidationClient);
}

void UserContentProvider::unregisterForUserMessageHandlerInvalidation(UserContentProviderInvalidationClient& invalidationClient)
{
    ASSERT(m_userMessageHandlerInvalidationClients.contains(invalidationClient));

    m_userMessageHandlerInvalidationClients.remove(invalidationClient);
}

void UserContentProvider::invalidateAllRegisteredUserMessageHandlerInvalidationClients()
{
    for (auto& client : m_userMessageHandlerInvalidationClients)
        client.didInvalidate(*this);
}

void UserContentProvider::invalidateInjectedStyleSheetCacheInAllFramesInAllPages()
{
    for (auto& page : m_pages)
        page.invalidateInjectedStyleSheetCacheInAllFrames();
}

#if ENABLE(CONTENT_EXTENSIONS)
static DocumentLoader* mainDocumentLoader(DocumentLoader& loader)
{
    if (auto frame = loader.frame()) {
        if (frame->isMainFrame())
            return &loader;

        auto* localFrame = dynamicDowncast<LocalFrame>(frame->mainFrame());
        if (localFrame)
            return localFrame->loader().documentLoader();
    }
    return nullptr;
}

static ContentExtensions::ContentExtensionsBackend::RuleListFilter ruleListFilter(DocumentLoader& documentLoader)
{
    RefPtr mainLoader = mainDocumentLoader(documentLoader);
    if (!mainLoader) {
        return [](const String&) {
            return ContentExtensions::ContentExtensionsBackend::ShouldSkipRuleList::No;
        };
    }

    auto policySourceLoader = mainLoader;
    if (!mainLoader->request().url().hasSpecialScheme() && documentLoader.request().url().protocolIsInHTTPFamily())
        policySourceLoader = &documentLoader;

    auto& exceptions = policySourceLoader->contentExtensionEnablement().second;
    switch (policySourceLoader->contentExtensionEnablement().first) {
    case ContentExtensionDefaultEnablement::Disabled:
        return [&](auto& identifier) {
            return exceptions.contains(identifier)
                ? ContentExtensions::ContentExtensionsBackend::ShouldSkipRuleList::No
                : ContentExtensions::ContentExtensionsBackend::ShouldSkipRuleList::Yes;
        };
    case ContentExtensionDefaultEnablement::Enabled:
        return [&](auto& identifier) {
            return exceptions.contains(identifier)
                ? ContentExtensions::ContentExtensionsBackend::ShouldSkipRuleList::Yes
                : ContentExtensions::ContentExtensionsBackend::ShouldSkipRuleList::No;
        };
    }
    ASSERT_NOT_REACHED();
    return { };
}

static void applyLinkDecorationFilteringIfNeeded(ContentRuleListResults& results, Page& page, const URL& url, const DocumentLoader& initiatingDocumentLoader)
{
    if (RefPtr frame = initiatingDocumentLoader.frame(); !frame || !frame->isMainFrame())
        return;

    if (auto adjustedURL = page.chrome().client().applyLinkDecorationFiltering(url, LinkDecorationFilteringTrigger::Navigation); adjustedURL != url)
        results.summary.redirectActions.append({ { ContentExtensions::RedirectAction::URLAction { adjustedURL.string() } }, adjustedURL });
}

ContentRuleListResults UserContentProvider::processContentRuleListsForLoad(Page& page, const URL& url, OptionSet<ContentExtensions::ResourceType> resourceType, DocumentLoader& initiatingDocumentLoader, const URL& redirectFrom)
{
    auto results = userContentExtensionBackend().processContentRuleListsForLoad(page, url, resourceType, initiatingDocumentLoader, redirectFrom, ruleListFilter(initiatingDocumentLoader));

    if (resourceType.contains(ContentExtensions::ResourceType::Document))
        applyLinkDecorationFilteringIfNeeded(results, page, url, initiatingDocumentLoader);

    return results;
}
#endif // ENABLE(CONTENT_EXTENSIONS)

} // namespace WebCore
