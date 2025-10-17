/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 6, 2024.
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

#include "CachedFrame.h"
#include <wtf/CheckedRef.h>
#include <wtf/MonotonicTime.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakRef.h>

namespace WebCore {

class Document;
class DocumentLoader;
class Page;

class CachedPage final : public CanMakeCheckedPtr<CachedPage> {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(CachedPage, WEBCORE_EXPORT);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(CachedPage);
public:
    explicit CachedPage(Page&);
    WEBCORE_EXPORT ~CachedPage();

    WEBCORE_EXPORT void restore(Page&);
    void clear();

    Page& page() const { return m_page.get(); }
    Document* document() const { return m_cachedMainFrame->document(); }
    DocumentLoader* documentLoader() const { return m_cachedMainFrame->documentLoader(); }
    RefPtr<DocumentLoader> protectedDocumentLoader() const;

    bool hasExpired() const;
    
    CachedFrame* cachedMainFrame() const { return m_cachedMainFrame.get(); }

#if ENABLE(VIDEO)
    void markForCaptionPreferencesChanged() { m_needsCaptionPreferencesChanged = true; }
#endif

    void markForDeviceOrPageScaleChanged() { m_needsDeviceOrPageScaleChanged = true; }

    void markForContentsSizeChanged() { m_needsUpdateContentsSize = true; }

private:
    WeakRef<Page> m_page;
    MonotonicTime m_expirationTime;
    std::unique_ptr<CachedFrame> m_cachedMainFrame;
#if ENABLE(VIDEO)
    bool m_needsCaptionPreferencesChanged { false };
#endif
    bool m_needsDeviceOrPageScaleChanged { false };
    bool m_needsUpdateContentsSize { false };
    Vector<RegistrableDomain> m_loadedSubresourceDomains;
};

} // namespace WebCore
