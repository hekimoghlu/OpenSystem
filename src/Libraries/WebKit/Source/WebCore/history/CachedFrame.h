/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 29, 2024.
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

#include "LocalDOMWindow.h"
#include <wtf/URL.h>
#include "ScriptCachedFrameData.h"
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/UniqueRef.h>

namespace WebCore {

class CachedFrame;
class CachedFramePlatformData;
class Document;
class DocumentLoader;
class LocalFrameView;
class Node;
enum class HasInsecureContent : bool;
enum class UsedLegacyTLS : bool;
enum class WasPrivateRelayed : bool;

class CachedFrameBase {
public:
    void restore();

    Document* document() const { return m_document.get(); }
    FrameView* view() const { return m_view.get(); }
    RefPtr<FrameView> protectedView() const;
    const URL& url() const { return m_url; }
    bool isMainFrame() { return m_isMainFrame; }

protected:
    CachedFrameBase(Frame&);
    ~CachedFrameBase();

    void pruneDetachedChildFrames();

    RefPtr<Document> m_document;
    RefPtr<DocumentLoader> m_documentLoader;
    RefPtr<FrameView> m_view;
    URL m_url;
    std::unique_ptr<ScriptCachedFrameData> m_cachedFrameScriptData;
    std::unique_ptr<CachedFramePlatformData> m_cachedFramePlatformData;
    bool m_isMainFrame;

    Vector<UniqueRef<CachedFrame>> m_childFrames;

private:
    void initializeWithLocalFrame(LocalFrame&);
};

class CachedFrame : private CachedFrameBase {
    WTF_MAKE_TZONE_ALLOCATED(CachedFrame);
public:
    explicit CachedFrame(Frame&);

    void open();
    void clear();
    void destroy();

    WEBCORE_EXPORT void setCachedFramePlatformData(std::unique_ptr<CachedFramePlatformData>);
    WEBCORE_EXPORT CachedFramePlatformData* cachedFramePlatformData();

    HasInsecureContent hasInsecureContent() const;
    UsedLegacyTLS usedLegacyTLS() const;
    WasPrivateRelayed wasPrivateRelayed() const;

    using CachedFrameBase::document;
    using CachedFrameBase::view;
    using CachedFrameBase::url;
    DocumentLoader* documentLoader() const { return m_documentLoader.get(); }

    size_t descendantFrameCount() const;
};

} // namespace WebCore
