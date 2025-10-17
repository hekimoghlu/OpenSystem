/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 14, 2025.
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

#if USE(GSTREAMER) && ENABLE(VIDEO)

#include "GStreamerCommon.h"
#include "GStreamerElementHarness.h"
#include "ImageDecoder.h"
#include "MIMETypeRegistry.h"
#include "SampleMap.h"
#include "SharedBuffer.h"
#include <wtf/Forward.h>
#include <wtf/Lock.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class ContentType;
class ImageDecoderGStreamerSample;

void teardownGStreamerImageDecoders();

class ImageDecoderGStreamer final : public ImageDecoder {
    WTF_MAKE_TZONE_ALLOCATED(ImageDecoderGStreamer);
    WTF_MAKE_NONCOPYABLE(ImageDecoderGStreamer);
public:
    static RefPtr<ImageDecoderGStreamer> create(FragmentedSharedBuffer&, const String& mimeType, AlphaOption, GammaAndColorProfileOption);
    ImageDecoderGStreamer(FragmentedSharedBuffer&, const String& mimeType, AlphaOption, GammaAndColorProfileOption);
    ~ImageDecoderGStreamer();

    static bool supportsMediaType(MediaType type) { return type == MediaType::Video; }
    static bool supportsContainerType(const String&);

    size_t bytesDecodedToDetermineProperties() const override { return 0; }
    static bool canDecodeType(const String& mimeType);

    void setEncodedDataStatusChangeCallback(Function<void(EncodedDataStatus)>&& callback) final { m_encodedDataStatusChangedCallback = WTFMove(callback); }
    EncodedDataStatus encodedDataStatus() const final;
    IntSize size() const final;
    size_t frameCount() const final { return m_sampleData.size(); }
    RepetitionCount repetitionCount() const final;
    String filenameExtension() const final { return MIMETypeRegistry::preferredExtensionForMIMEType(m_mimeType); }
    std::optional<IntPoint> hotSpot() const final { return std::nullopt; }

    IntSize frameSizeAtIndex(size_t, SubsamplingLevel = SubsamplingLevel::Default) const final { return size(); }
    bool frameIsCompleteAtIndex(size_t index) const final { return sampleAtIndex(index); }

    Seconds frameDurationAtIndex(size_t) const final;
    bool frameHasAlphaAtIndex(size_t) const final;
    unsigned frameBytesAtIndex(size_t, SubsamplingLevel = SubsamplingLevel::Default) const final;

    PlatformImagePtr createFrameImageAtIndex(size_t, SubsamplingLevel = SubsamplingLevel::Default, const DecodingOptions& = DecodingOptions(DecodingMode::Synchronous)) final;

    void setExpectedContentSize(long long) final { }
    void setData(const FragmentedSharedBuffer&, bool allDataReceived) final;
    bool isAllDataReceived() const final { return m_eos; }
    void clearFrameBufferCache(size_t) final;

    void tearDown();

private:
    void pushEncodedData(const FragmentedSharedBuffer&);
    void storeDecodedSample(GRefPtr<GstSample>&&);
    const ImageDecoderGStreamerSample* sampleAtIndex(size_t) const;

    Function<void(EncodedDataStatus)> m_encodedDataStatusChangedCallback;
    SampleMap m_sampleData;
    DecodeOrderSampleMap::iterator m_cursor;
    Lock m_sampleGeneratorLock;
    bool m_eos { false };
    bool m_error { false };
    std::optional<IntSize> m_size;
    String m_mimeType;

    RefPtr<GStreamerElementHarness> m_parserHarness;
    RefPtr<GStreamerElementHarness> m_decoderHarness;
};

} // namespace WebCore

#endif // USE(GSTREAMER) && ENABLE(VIDEO)
