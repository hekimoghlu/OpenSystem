/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 21, 2022.
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

#include "CachedResourceHandle.h"
#include "Document.h"
#include "Timer.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/WeakRef.h>

namespace WebCore {

class CachedFont;

class DocumentFontLoader {
    WTF_MAKE_TZONE_ALLOCATED(DocumentFontLoader);
public:
    DocumentFontLoader(Document&);
    ~DocumentFontLoader();

    void ref() const;
    void deref() const;

    CachedFont* cachedFont(URL&&, bool, bool, LoadedFromOpaqueSource);
    void beginLoadingFontSoon(CachedFont&);

    void loadPendingFonts();
    void stopLoadingAndClearFonts();

    void suspendFontLoading();
    void resumeFontLoading();

private:
    void fontLoadingTimerFired();

    WeakRef<Document, WeakPtrImplWithEventTargetData> m_document;
    Timer m_fontLoadingTimer;
    Vector<CachedResourceHandle<CachedFont>> m_fontsToBeginLoading;
    bool m_isFontLoadingSuspended { false };
    bool m_isStopped { false };
};

} // namespace WebCore
