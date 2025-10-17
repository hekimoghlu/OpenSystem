/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 20, 2024.
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
#include "XSLImportRule.h"

#if ENABLE(XSLT)

#include "CachedXSLStyleSheet.h"
#include "CachedResourceLoader.h"
#include "CachedResourceRequest.h"
#include "Document.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(XSLImportRule);

XSLImportRule::XSLImportRule(XSLStyleSheet& parent, const String& href)
    : m_parentStyleSheet(parent)
    , m_strHref(href)
{
}

XSLImportRule::~XSLImportRule()
{
    if (m_styleSheet)
        m_styleSheet->setParentStyleSheet(nullptr);

    if (m_cachedSheet)
        m_cachedSheet->removeClient(*this);
}

void XSLImportRule::setXSLStyleSheet(const String& href, const URL& baseURL, const String& sheet)
{
    if (m_styleSheet)
        m_styleSheet->setParentStyleSheet(nullptr);

    // FIXME: parentStyleSheet() should never be null here.
    RefPtr parent = parentStyleSheet();
    m_styleSheet = XSLStyleSheet::create(parent.get(), href, baseURL);

    m_styleSheet->parseString(sheet);
    m_loading = false;

    if (parent)
        parent->checkLoaded();
}

bool XSLImportRule::isLoading()
{
    return (m_loading || (m_styleSheet && m_styleSheet->isLoading()));
}

void XSLImportRule::loadSheet()
{
    RefPtr rootSheet = parentStyleSheet();
    while (auto* parentSheet = rootSheet->parentStyleSheet())
        rootSheet = parentSheet;

    RefPtr cachedResourceLoader = rootSheet->cachedResourceLoader();

    String absHref = m_strHref;
    RefPtr parentSheet = parentStyleSheet();
    if (!parentSheet->baseURL().isNull())
        // use parent styleheet's URL as the base URL
        absHref = URL(parentSheet->baseURL(), m_strHref).string();

    // Check for a cycle in our import chain.  If we encounter a stylesheet
    // in our parent chain with the same URL, then just bail.
    for (RefPtr parentSheet = parentStyleSheet(); parentSheet; parentSheet = parentSheet->parentStyleSheet()) {
        if (absHref == parentSheet->baseURL().string())
            return;
    }

    if (m_cachedSheet)
        m_cachedSheet->removeClient(*this);

    auto options = CachedResourceLoader::defaultCachedResourceOptions();
    options.mode = FetchOptions::Mode::SameOrigin;
    m_cachedSheet = cachedResourceLoader->requestXSLStyleSheet({ResourceRequest(cachedResourceLoader->document()->completeURL(absHref)), options}).value_or(nullptr);

    if (m_cachedSheet) {
        m_cachedSheet->addClient(*this);

        // If the imported sheet is in the cache, then setXSLStyleSheet gets called,
        // and the sheet even gets parsed (via parseString).  In this case we have
        // loaded (even if our subresources haven't), so if we have a stylesheet after
        // checking the cache, then we've clearly loaded.
        if (!m_styleSheet)
            m_loading = true;
    }
}

} // namespace WebCore

#endif // ENABLE(XSLT)
