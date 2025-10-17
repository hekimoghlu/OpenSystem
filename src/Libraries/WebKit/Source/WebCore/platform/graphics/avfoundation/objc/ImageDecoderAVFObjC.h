/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 15, 2022.
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

#if HAVE(AVASSETREADER)

#include "ImageDecoder.h"
#include "ProcessIdentity.h"
#include "SampleMap.h"
#include <wtf/Lock.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

OBJC_CLASS AVAssetTrack;
OBJC_CLASS AVAssetReader;
OBJC_CLASS AVURLAsset;
OBJC_CLASS WebCoreSharedBufferResourceLoaderDelegate;
typedef struct opaqueCMSampleBuffer* CMSampleBufferRef;

namespace WTF {
class MediaTime;
}

namespace WebCore {

class ContentType;
class ImageDecoderAVFObjCSample;
class ImageRotationSessionVT;
class PixelBufferConformerCV;
class WebCoreDecompressionSession;

class ImageDecoderAVFObjC : public ImageDecoder {
    WTF_MAKE_TZONE_ALLOCATED(ImageDecoderAVFObjC);
public:
    WEBCORE_EXPORT static RefPtr<ImageDecoderAVFObjC> create(const FragmentedSharedBuffer&, const String& mimeType, AlphaOption, GammaAndColorProfileOption, ProcessIdentity resourceOwner);
    virtual ~ImageDecoderAVFObjC();

    WEBCORE_EXPORT static bool supportsMediaType(MediaType);
    static bool supportsContainerType(const String&);

    size_t bytesDecodedToDetermineProperties() const override { return 0; }
    WEBCORE_EXPORT static bool canDecodeType(const String& mimeType);

    const String& mimeType() const { return m_mimeType; }

    WEBCORE_EXPORT void setEncodedDataStatusChangeCallback(Function<void(EncodedDataStatus)>&&) final;
    EncodedDataStatus encodedDataStatus() const final;
    WEBCORE_EXPORT IntSize size() const final;
    WEBCORE_EXPORT size_t frameCount() const final;
    RepetitionCount repetitionCount() const final;
    String uti() const final;
    String filenameExtension() const final;
    std::optional<IntPoint> hotSpot() const final { return std::nullopt; }
    String accessibilityDescription() const final { return String(); }

    IntSize frameSizeAtIndex(size_t, SubsamplingLevel = SubsamplingLevel::Default) const final;
    bool frameIsCompleteAtIndex(size_t) const final;

    Seconds frameDurationAtIndex(size_t) const final;
    bool frameHasAlphaAtIndex(size_t) const final;
    unsigned frameBytesAtIndex(size_t, SubsamplingLevel = SubsamplingLevel::Default) const final;

    WEBCORE_EXPORT PlatformImagePtr createFrameImageAtIndex(size_t, SubsamplingLevel = SubsamplingLevel::Default, const DecodingOptions& = DecodingOptions(DecodingMode::Synchronous)) final;

    WEBCORE_EXPORT void setExpectedContentSize(long long) final;
    WEBCORE_EXPORT void setData(const FragmentedSharedBuffer&, bool allDataReceived) final;
    bool isAllDataReceived() const final { return m_isAllDataReceived; }
    WEBCORE_EXPORT void clearFrameBufferCache(size_t) final;

    bool hasTrack() const { return !!m_track; }
    WEBCORE_EXPORT Vector<ImageDecoder::FrameInfo> frameInfos() const;

private:
    ImageDecoderAVFObjC(const FragmentedSharedBuffer&, const String& mimeType, AlphaOption, GammaAndColorProfileOption, ProcessIdentity resourceOwner);

    AVAssetTrack *firstEnabledTrack();
    void readSamples();
    void readTrackMetadata();
    bool storeSampleBuffer(CMSampleBufferRef);
    void advanceCursor();
    void setTrack(AVAssetTrack *);

    const ImageDecoderAVFObjCSample* sampleAtIndex(size_t) const;
    bool sampleIsComplete(const ImageDecoderAVFObjCSample&) const;

    String m_mimeType;
    String m_uti;
    RetainPtr<AVURLAsset> m_asset;
    RetainPtr<AVAssetTrack> m_track;
    RetainPtr<WebCoreSharedBufferResourceLoaderDelegate> m_loader;
    std::unique_ptr<ImageRotationSessionVT> m_imageRotationSession;
    Ref<WebCoreDecompressionSession> m_decompressionSession;
    Function<void(EncodedDataStatus)> m_encodedDataStatusChangedCallback;

    SampleMap m_sampleData;
    DecodeOrderSampleMap::iterator m_cursor;
    Lock m_sampleGeneratorLock;
    bool m_isAllDataReceived { false };
    std::optional<IntSize> m_size;
    ProcessIdentity m_resourceOwner;
};

}
#endif
