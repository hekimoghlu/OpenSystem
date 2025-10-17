/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 28, 2022.
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

#if ENABLE(VIDEO) && USE(AVFOUNDATION)

#include <wtf/Forward.h>
#include <wtf/Noncopyable.h>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

OBJC_CLASS AVAssetResourceLoadingRequest;

namespace WebCore {

class CachedResourceMediaLoader;
class DataURLResourceMediaLoader;
class FragmentedSharedBuffer;
class MediaPlayerPrivateAVFoundationObjC;
class ParsedContentRange;
class PlatformResourceMediaLoader;
class ResourceError;
class ResourceResponse;

class WebCoreAVFResourceLoader : public ThreadSafeRefCounted<WebCoreAVFResourceLoader> {
    WTF_MAKE_TZONE_ALLOCATED(WebCoreAVFResourceLoader);
    WTF_MAKE_NONCOPYABLE(WebCoreAVFResourceLoader);
public:
    static Ref<WebCoreAVFResourceLoader> create(MediaPlayerPrivateAVFoundationObjC* parent, AVAssetResourceLoadingRequest*, GuaranteedSerialFunctionDispatcher&);
    virtual ~WebCoreAVFResourceLoader();

    void startLoading();
    void stopLoading();

private:
    WebCoreAVFResourceLoader(MediaPlayerPrivateAVFoundationObjC* parent, AVAssetResourceLoadingRequest*, GuaranteedSerialFunctionDispatcher&);

    friend class CachedResourceMediaLoader;
    friend class DataURLResourceMediaLoader;
    friend class PlatformResourceMediaLoader;

    // Return true if stopLoading() got called, indicating that no further processing should occur.
    bool responseReceived(const String&, int, const ParsedContentRange&, size_t);
    bool newDataStoredInSharedBuffer(const FragmentedSharedBuffer&);

    void loadFailed(const ResourceError&);
    void loadFinished();

    ThreadSafeWeakPtr<MediaPlayerPrivateAVFoundationObjC> m_parent;
    RetainPtr<AVAssetResourceLoadingRequest> m_avRequest;
    std::unique_ptr<DataURLResourceMediaLoader> m_dataURLMediaLoader;
    std::unique_ptr<CachedResourceMediaLoader> m_resourceMediaLoader;
    RefPtr<PlatformResourceMediaLoader> m_platformMediaLoader;
    bool m_isBlob { false };
    size_t m_responseOffset { 0 };
    int64_t m_requestedLength { 0 };
    int64_t m_requestedOffset { 0 };
    int64_t m_currentOffset { 0 };

    Ref<GuaranteedSerialFunctionDispatcher> m_targetDispatcher;
};

}

#endif // ENABLE(VIDEO) && USE(AVFOUNDATION)
