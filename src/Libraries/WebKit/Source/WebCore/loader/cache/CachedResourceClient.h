/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 27, 2025.
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

#include "FrameLoaderTypes.h"
#include <wtf/Noncopyable.h>
#include <wtf/WeakHashSet.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class CachedResourceClient;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::CachedResourceClient> : std::true_type { };
}

namespace WebCore {

class CachedResource;
class NetworkLoadMetrics;

class WEBCORE_EXPORT CachedResourceClient : public CanMakeSingleThreadWeakPtr<CachedResourceClient> {
    WTF_MAKE_NONCOPYABLE(CachedResourceClient);
public:
    enum CachedResourceClientType {
        BaseResourceType,
        ImageType,
        FontType,
        StyleSheetType,
        SVGDocumentType,
        RawResourceType
    };

    virtual ~CachedResourceClient();

    virtual void notifyFinished(CachedResource&, const NetworkLoadMetrics&, LoadWillContinueInAnotherProcess = LoadWillContinueInAnotherProcess::No);
    virtual void deprecatedDidReceiveCachedResource(CachedResource&);

    static CachedResourceClientType expectedType();
    virtual CachedResourceClientType resourceClientType() const;
    virtual bool shouldMarkAsReferenced() const;

#if ASSERT_ENABLED
    void addAssociatedResource(CachedResource&);
    void removeAssociatedResource(CachedResource&);
#endif

protected:
    CachedResourceClient();

private:
#if ASSERT_ENABLED
    WeakHashSet<CachedResource> m_associatedResources;
#endif
};

} // namespace WebCore

#define SPECIALIZE_TYPE_TRAITS_CACHED_RESOURCE_CLIENT(ToClassName, CachedResourceTypeValue) \
SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::ToClassName) \
    static bool isType(const WebCore::CachedResourceClient& client) { return client.resourceClientType() == WebCore::CachedResourceClient::CachedResourceTypeValue; } \
SPECIALIZE_TYPE_TRAITS_END()
