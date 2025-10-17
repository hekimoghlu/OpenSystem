/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 2, 2024.
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

#if ENABLE(VIDEO) && USE(AVFOUNDATION) && HAVE(AVFOUNDATION_LOADER_DELEGATE)

#include "CachedRawResourceClient.h"
#include "CachedResourceHandle.h"
#include <wtf/Noncopyable.h>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>

typedef struct OpaqueAVCFAssetResourceLoadingRequest* AVCFAssetResourceLoadingRequestRef;

namespace WebCore {

class CachedRawResource;
class CachedResourceLoader;
class MediaPlayerPrivateAVFoundationCF;

class WebCoreAVCFResourceLoader : public RefCounted<WebCoreAVCFResourceLoader>, CachedRawResourceClient {
    WTF_MAKE_TZONE_ALLOCATED(WebCoreAVCFResourceLoader);
    WTF_MAKE_NONCOPYABLE(WebCoreAVCFResourceLoader);
public:
    static Ref<WebCoreAVCFResourceLoader> create(MediaPlayerPrivateAVFoundationCF* parent, AVCFAssetResourceLoadingRequestRef);
    virtual ~WebCoreAVCFResourceLoader();

    void startLoading();
    void stopLoading();
    void invalidate();

    CachedRawResource* resource();

private:
    // CachedRawResourceClient
    void responseReceived(CachedResource&, const ResourceResponse&, CompletionHandler<void()>&&) override;
    void dataReceived(CachedResource&, const SharedBuffer&) override;
    void notifyFinished(CachedResource&, const NetworkLoadMetrics&) override;

    void fulfillRequestWithResource(CachedResource&);

    WebCoreAVCFResourceLoader(MediaPlayerPrivateAVFoundationCF* parent, AVCFAssetResourceLoadingRequestRef);
    MediaPlayerPrivateAVFoundationCF* m_parent;
    RetainPtr<AVCFAssetResourceLoadingRequestRef> m_avRequest;
    CachedResourceHandle<CachedRawResource> m_resource;
};

}

#endif // ENABLE(VIDEO) && USE(AVFOUNDATION)
