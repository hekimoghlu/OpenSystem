/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 13, 2022.
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

#if ENABLE(VIDEO)

#include "CachedResourceClient.h"
#include "CachedResourceHandle.h"
#include "LoaderMalloc.h"
#include "Timer.h"
#include "WebVTTParser.h"
#include <memory>
#include <wtf/CheckedPtr.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class TextTrackLoaderClient;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::TextTrackLoaderClient> : std::true_type { };
}

namespace WebCore {

class CachedTextTrack;
class Document;
class HTMLTrackElement;
class TextTrackLoader;
class VTTCue;

class TextTrackLoaderClient : public CanMakeWeakPtr<TextTrackLoaderClient> {
public:
    virtual ~TextTrackLoaderClient() = default;
    
    virtual void newCuesAvailable(TextTrackLoader&) = 0;
    virtual void cueLoadingCompleted(TextTrackLoader&, bool loadingFailed) = 0;
    virtual void newRegionsAvailable(TextTrackLoader&) = 0;
    virtual void newStyleSheetsAvailable(TextTrackLoader&) = 0;
};

class TextTrackLoader final : public CachedResourceClient, private WebVTTParserClient, public CanMakeCheckedPtr<TextTrackLoader> {
    WTF_MAKE_NONCOPYABLE(TextTrackLoader); 
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(Loader);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(TextTrackLoader);
public:
    TextTrackLoader(TextTrackLoaderClient&, Document&);
    virtual ~TextTrackLoader();

    bool load(const URL&, HTMLTrackElement&);
    void cancelLoad();

    Vector<Ref<VTTCue>> getNewCues();
    Vector<Ref<VTTRegion>> getNewRegions();
    Vector<String> getNewStyleSheets();

private:
    // CachedResourceClient
    void notifyFinished(CachedResource&, const NetworkLoadMetrics&, LoadWillContinueInAnotherProcess) final;
    void deprecatedDidReceiveCachedResource(CachedResource&) final;

    // WebVTTParserClient
    void newCuesParsed() final;
    void newRegionsParsed() final;
    void newStyleSheetsParsed() final;
    void fileFailedToParse() final;

    void processNewCueData(CachedResource&);
    void cueLoadTimerFired();
    void corsPolicyPreventedLoad();

    Ref<Document> protectedDocument() const;
    CachedResourceHandle<CachedTextTrack> protectedResource() const;

    enum State { Idle, Loading, Finished, Failed };

    WeakRef<TextTrackLoaderClient> m_client;
    std::unique_ptr<WebVTTParser> m_cueParser;
    CachedResourceHandle<CachedTextTrack> m_resource;
    WeakRef<Document, WeakPtrImplWithEventTargetData> m_document;
    Timer m_cueLoadTimer;
    State m_state { Idle };
    unsigned m_parseOffset { 0 };
    bool m_newCuesAvailable { false };
};

} // namespace WebCore

#endif // ENABLE(VIDEO)
