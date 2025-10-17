/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 27, 2024.
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

#if ENABLE(PREVIEW_CONVERTER)

#include "ResourceError.h"
#include "ResourceResponse.h"
#include "SharedBuffer.h"
#include <wtf/RefCounted.h>
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

OBJC_CLASS QLPreviewConverter;
OBJC_CLASS WebPreviewConverterDelegate;

namespace WebCore {
struct PreviewPlatformDelegate;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::PreviewPlatformDelegate> : std::true_type { };
}

namespace WebCore {

class ResourceError;
class ResourceRequest;
struct PreviewConverterClient;
struct PreviewConverterProvider;

struct PreviewPlatformDelegate : CanMakeWeakPtr<PreviewPlatformDelegate> {
    virtual ~PreviewPlatformDelegate() = default;

    virtual void delegateDidReceiveData(const FragmentedSharedBuffer&) = 0;
    virtual void delegateDidFinishLoading() = 0;
    virtual void delegateDidFailWithError(const ResourceError&) = 0;
};

class PreviewConverter final : private PreviewPlatformDelegate, public RefCounted<PreviewConverter> {
    WTF_MAKE_TZONE_ALLOCATED(PreviewConverter);
    WTF_MAKE_NONCOPYABLE(PreviewConverter);
public:
    static Ref<PreviewConverter> create(const ResourceResponse& response, PreviewConverterProvider& provider)
    {
        return adoptRef(*new PreviewConverter(response, provider));
    }

    WEBCORE_EXPORT static bool supportsMIMEType(const String& mimeType);

    ~PreviewConverter();

    ResourceRequest safeRequest(const ResourceRequest&) const;
    ResourceResponse previewResponse() const;
    WEBCORE_EXPORT String previewFileName() const;
    WEBCORE_EXPORT String previewUTI() const;
    const ResourceError& previewError() const;
    const FragmentedSharedBuffer& previewData() const;

    void failedUpdating();
    void finishUpdating();
    void updateMainResource();

    bool hasClient(PreviewConverterClient&) const;
    void addClient(PreviewConverterClient&);
    void removeClient(PreviewConverterClient&);

    WEBCORE_EXPORT static const String& passwordForTesting();
    WEBCORE_EXPORT static void setPasswordForTesting(const String&);

private:
    static UncheckedKeyHashSet<String, ASCIICaseInsensitiveHash> platformSupportedMIMETypes();

    PreviewConverter(const ResourceResponse&, PreviewConverterProvider&);

    ResourceResponse platformPreviewResponse() const;
    bool isPlatformPasswordError(const ResourceError&) const;

    template<typename T> void iterateClients(T&& callback);
    void appendFromBuffer(const FragmentedSharedBuffer&);
    void didAddClient(PreviewConverterClient&);
    void didFailConvertingWithError(const ResourceError&);
    void didFailUpdating();
    void replayToClient(PreviewConverterClient&);

    void platformAppend(const SharedBufferDataView&);
    void platformFailedAppending();
    void platformFinishedAppending();
    void platformUnlockWithPassword(const String&);

    // PreviewPlatformDelegate
    void delegateDidReceiveData(const FragmentedSharedBuffer&) final;
    void delegateDidFinishLoading() final;
    void delegateDidFailWithError(const ResourceError&) final;

    enum class State : uint8_t {
        Updating,
        FailedUpdating,
        Converting,
        FailedConverting,
        FinishedConverting,
    };

    SharedBufferBuilder m_previewData;
    ResourceError m_previewError;
    ResourceResponse m_originalResponse;
    State m_state { State::Updating };
    Vector<WeakPtr<PreviewConverterClient>, 1> m_clients;
    WeakPtr<PreviewConverterProvider> m_provider;
    bool m_isInClientCallback { false };
    size_t m_lengthAppended { 0 };

#if USE(QUICK_LOOK)
    RetainPtr<WebPreviewConverterDelegate> m_platformDelegate;
    RetainPtr<QLPreviewConverter> m_platformConverter;
#endif
};

} // namespace WebCore

#endif // ENABLE(PREVIEW_CONVERTER)
