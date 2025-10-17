/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 14, 2025.
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

#if ENABLE(GPU_PROCESS) && HAVE(AVASSETREADER)

#include "MessageReceiver.h"
#include "RemoteImageDecoderAVFManager.h"
#include <WebCore/ImageDecoder.h>
#include <WebCore/ImageDecoderIdentifier.h>
#include <wtf/Function.h>
#include <wtf/HashMap.h>
#include <wtf/WeakPtr.h>

namespace WebKit {

class GPUProcessConnection;
class WebProcess;

class RemoteImageDecoderAVF final
    : public WebCore::ImageDecoder
    , public CanMakeWeakPtr<RemoteImageDecoderAVF> {
public:
    static Ref<RemoteImageDecoderAVF> create(RemoteImageDecoderAVFManager& manager, const WebCore::ImageDecoderIdentifier& identifier, WebCore::FragmentedSharedBuffer&, const String& mimeType)
    {
        return adoptRef(*new RemoteImageDecoderAVF(manager, identifier, mimeType));
    }
    RemoteImageDecoderAVF(RemoteImageDecoderAVFManager&, const WebCore::ImageDecoderIdentifier&, const String& mimeType);

    virtual ~RemoteImageDecoderAVF();

    static bool canDecodeType(const String& mimeType);
    static bool supportsMediaType(MediaType);

    size_t bytesDecodedToDetermineProperties() const override { return 0; }

    WebCore::EncodedDataStatus encodedDataStatus() const final;
    void setEncodedDataStatusChangeCallback(WTF::Function<void(WebCore::EncodedDataStatus)>&&) final;
    WebCore::IntSize size() const final;
    size_t frameCount() const final;
    WebCore::RepetitionCount repetitionCount() const final;
    String uti() const final;
    String filenameExtension() const final;
    std::optional<WebCore::IntPoint> hotSpot() const final { return std::nullopt; }
    String accessibilityDescription() const final { return String(); }

    WebCore::IntSize frameSizeAtIndex(size_t, WebCore::SubsamplingLevel = WebCore::SubsamplingLevel::Default) const final;
    bool frameIsCompleteAtIndex(size_t) const final;

    Seconds frameDurationAtIndex(size_t) const final;
    bool frameHasAlphaAtIndex(size_t) const final;
    unsigned frameBytesAtIndex(size_t, WebCore::SubsamplingLevel = WebCore::SubsamplingLevel::Default) const final;

    WebCore::PlatformImagePtr createFrameImageAtIndex(size_t, WebCore::SubsamplingLevel = WebCore::SubsamplingLevel::Default, const WebCore::DecodingOptions& = WebCore::DecodingOptions(WebCore::DecodingMode::Synchronous)) final;

    void setExpectedContentSize(long long) final;
    void setData(const WebCore::FragmentedSharedBuffer&, bool allDataReceived) final;
    bool isAllDataReceived() const final { return m_isAllDataReceived; }
    void clearFrameBufferCache(size_t) final;

    void encodedDataStatusChanged(size_t frameCount, const WebCore::IntSize&, bool hasTrack);

private:
    Ref<RemoteImageDecoderAVFManager> protectedManager() const;

    ThreadSafeWeakPtr<GPUProcessConnection> m_gpuProcessConnection;
    ThreadSafeWeakPtr<RemoteImageDecoderAVFManager> m_manager; // Cannot be null.
    WebCore::ImageDecoderIdentifier m_identifier;

    String m_mimeType;
    String m_uti;
    bool m_isAllDataReceived { false };
    WTF::Function<void(WebCore::EncodedDataStatus)> m_encodedDataStatusChangedCallback;
    HashMap<int, WebCore::PlatformImagePtr, WTF::IntHash<int>, WTF::UnsignedWithZeroKeyHashTraits<int>> m_frameImages;
    Vector<ImageDecoder::FrameInfo> m_frameInfos;
    size_t m_frameCount { 0 };
    std::optional<WebCore::IntSize> m_size;
    bool m_hasTrack { false };
};

}
#endif
