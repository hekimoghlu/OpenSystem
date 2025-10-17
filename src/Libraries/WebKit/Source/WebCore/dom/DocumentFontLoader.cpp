/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 23, 2023.
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
#include "DocumentFontLoader.h"

#include "CSSFontSelector.h"
#include "CachedFont.h"
#include "CachedResourceLoader.h"
#include "CachedResourceRequest.h"
#include "CachedResourceRequestInitiatorTypes.h"
#include "DocumentInlines.h"
#include "FrameDestructionObserverInlines.h"
#include "FrameLoader.h"
#include "LocalFrame.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(DocumentFontLoader);

DocumentFontLoader::DocumentFontLoader(Document& document)
    : m_document(document)
    , m_fontLoadingTimer(*this, &DocumentFontLoader::fontLoadingTimerFired)
{
}

DocumentFontLoader::~DocumentFontLoader()
{
    stopLoadingAndClearFonts();
}

void DocumentFontLoader::ref() const
{
    m_document->ref();
}

void DocumentFontLoader::deref() const
{
    m_document->deref();
}

CachedFont* DocumentFontLoader::cachedFont(URL&& url, bool isSVG, bool isInitiatingElementInUserAgentShadowTree, LoadedFromOpaqueSource loadedFromOpaqueSource)
{
    ResourceLoaderOptions options = CachedResourceLoader::defaultCachedResourceOptions();
    options.contentSecurityPolicyImposition = isInitiatingElementInUserAgentShadowTree ? ContentSecurityPolicyImposition::SkipPolicyCheck : ContentSecurityPolicyImposition::DoPolicyCheck;
    options.loadedFromOpaqueSource = loadedFromOpaqueSource;
    options.sameOriginDataURLFlag = SameOriginDataURLFlag::Set;

    CachedResourceRequest request(ResourceRequest(WTFMove(url)), options);
    request.setInitiatorType(cachedResourceRequestInitiatorTypes().css);
    return m_document->protectedCachedResourceLoader()->requestFont(WTFMove(request), isSVG).value_or(nullptr).get();
}

void DocumentFontLoader::beginLoadingFontSoon(CachedFont& font)
{
    if (m_isStopped)
        return;

    m_fontsToBeginLoading.append(&font);
    // Increment the request count now, in order to prevent didFinishLoad from being dispatched
    // after this font has been requested but before it began loading. Balanced by
    // decrementRequestCount() in fontLoadingTimerFired() and in stopLoadingAndClearFonts().
    m_document->protectedCachedResourceLoader()->incrementRequestCount(font);

    if (!m_isFontLoadingSuspended && !m_fontLoadingTimer.isActive())
        m_fontLoadingTimer.startOneShot(0_s);
}

void DocumentFontLoader::loadPendingFonts()
{
    if (m_isFontLoadingSuspended)
        return;

    Vector<CachedResourceHandle<CachedFont>> fontsToBeginLoading;
    fontsToBeginLoading.swap(m_fontsToBeginLoading);

    Ref cachedResourceLoader = m_document->cachedResourceLoader();
    for (auto& fontHandle : fontsToBeginLoading) {
        fontHandle->beginLoadIfNeeded(cachedResourceLoader);
        // Balances incrementRequestCount() in beginLoadingFontSoon().
        cachedResourceLoader->decrementRequestCount(*fontHandle);
    }
}

void DocumentFontLoader::fontLoadingTimerFired()
{
    Ref protectedThis { *this };
    loadPendingFonts();

    // FIXME: Use SubresourceLoader instead.
    // Call FrameLoader::loadDone before FrameLoader::subresourceLoadDone to match the order in SubresourceLoader::notifyDone.
    m_document->protectedCachedResourceLoader()->loadDone(LoadCompletionType::Finish);
    // Ensure that if the request count reaches zero, the frame loader will know about it.
    // New font loads may be triggered by layout after the document load is complete but before we have dispatched
    // didFinishLoading for the frame. Make sure the delegate is always dispatched by checking explicitly.
    if (RefPtr frame = m_document->frame())
        frame->protectedLoader()->checkLoadComplete();
}

void DocumentFontLoader::stopLoadingAndClearFonts()
{
    if (m_isStopped)
        return;

    m_fontLoadingTimer.stop();
    Ref cachedResourceLoader = m_document->cachedResourceLoader();
    for (auto& fontHandle : m_fontsToBeginLoading) {
        // Balances incrementRequestCount() in beginLoadingFontSoon().
        cachedResourceLoader->decrementRequestCount(*fontHandle);
    }
    m_fontsToBeginLoading.clear();
    if (RefPtr fontSelector = m_document->fontSelectorIfExists())
        fontSelector->clearFonts();

    m_isFontLoadingSuspended = true;
    m_isStopped = true;
}

void DocumentFontLoader::suspendFontLoading()
{
    if (m_isFontLoadingSuspended)
        return;

    m_fontLoadingTimer.stop();
    m_isFontLoadingSuspended = true;
}

void DocumentFontLoader::resumeFontLoading()
{
    if (!m_isFontLoadingSuspended || m_isStopped)
        return;

    m_isFontLoadingSuspended = false;
    if (!m_fontsToBeginLoading.isEmpty())
        m_fontLoadingTimer.startOneShot(0_s);
}

} // namespace WebCore
