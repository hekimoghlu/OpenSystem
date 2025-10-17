/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 25, 2023.
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

#include "CachedResource.h"
#include "CachedResourceRequest.h"
#include "ScriptType.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakRef.h>

namespace WebCore {
class HTMLResourcePreloader;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::HTMLResourcePreloader> : std::true_type { };
}

namespace WebCore {

class PreloadRequest {
    WTF_MAKE_TZONE_ALLOCATED(PreloadRequest);
public:
    PreloadRequest(ASCIILiteral initiatorType, const String& resourceURL, const URL& baseURL, CachedResource::Type resourceType, const String& mediaAttribute, ScriptType scriptType, const ReferrerPolicy& referrerPolicy, RequestPriority fetchPriority = RequestPriority::Auto)
        : m_initiatorType(initiatorType)
        , m_resourceURL(resourceURL)
        , m_baseURL(baseURL.isolatedCopy())
        , m_resourceType(resourceType)
        , m_mediaAttribute(mediaAttribute)
        , m_scriptType(scriptType)
        , m_referrerPolicy(referrerPolicy)
        , m_fetchPriority(fetchPriority)
    {
    }

    CachedResourceRequest resourceRequest(Document&);

    const String& charset() const { return m_charset; }
    const String& media() const { return m_mediaAttribute; }
    void setCharset(const String& charset) { m_charset = charset.isolatedCopy(); }
    void setCrossOriginMode(const String& mode) { m_crossOriginMode = mode; }
    void setNonce(const String& nonce) { m_nonceAttribute = nonce; }
    void setScriptIsAsync(bool value) { m_scriptIsAsync = value; }
    CachedResource::Type resourceType() const { return m_resourceType; }

private:
    URL completeURL(Document&);

    ASCIILiteral m_initiatorType;
    String m_resourceURL;
    URL m_baseURL;
    String m_charset;
    CachedResource::Type m_resourceType;
    String m_mediaAttribute;
    String m_crossOriginMode;
    String m_nonceAttribute;
    bool m_scriptIsAsync { false };
    ScriptType m_scriptType;
    ReferrerPolicy m_referrerPolicy;
    RequestPriority m_fetchPriority;
};

typedef Vector<std::unique_ptr<PreloadRequest>> PreloadRequestStream;

class HTMLResourcePreloader : public CanMakeWeakPtr<HTMLResourcePreloader> {
    WTF_MAKE_TZONE_ALLOCATED(HTMLResourcePreloader);
    WTF_MAKE_NONCOPYABLE(HTMLResourcePreloader);
public:
    explicit HTMLResourcePreloader(Document& document)
        : m_document(document)
    {
    }

    void preload(PreloadRequestStream);
    void preload(std::unique_ptr<PreloadRequest>);

private:
    Ref<Document> protectedDocument() const { return m_document.get(); }

    WeakRef<Document, WeakPtrImplWithEventTargetData> m_document;
};

} // namespace WebCore
