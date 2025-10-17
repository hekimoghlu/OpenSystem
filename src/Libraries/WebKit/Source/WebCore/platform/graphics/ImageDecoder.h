/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 7, 2022.
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

#include "DecodingOptions.h"
#include "ImageOrientation.h"
#include "ImageTypes.h"
#include "IntPoint.h"
#include "IntSize.h"
#include "PlatformImage.h"
#include <wtf/Seconds.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/WTFString.h>
#include <wtf/ThreadSafeRefCounted.h>

namespace WebCore {

class FragmentedSharedBuffer;
class ImageFrame;

struct ImageDecoderFrameInfo {
    bool hasAlpha;
    Seconds duration;
};

class ImageDecoder : public ThreadSafeRefCounted<ImageDecoder> {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(ImageDecoder, WEBCORE_EXPORT);
public:
    static RefPtr<ImageDecoder> create(FragmentedSharedBuffer&, const String& mimeType, AlphaOption, GammaAndColorProfileOption);
    WEBCORE_EXPORT virtual ~ImageDecoder();

    using FrameInfo = ImageDecoderFrameInfo;

    enum class MediaType {
        Image,
        Video,
    };

    static bool supportsMediaType(MediaType);

#if ENABLE(GPU_PROCESS)
    using SupportsMediaTypeFunc = Function<bool(MediaType)>;
    using CanDecodeTypeFunc = Function<bool(const String&)>;
    using CreateImageDecoderFunc = Function<RefPtr<ImageDecoder>(FragmentedSharedBuffer&, const String&, AlphaOption, GammaAndColorProfileOption)>;

    struct ImageDecoderFactory {
        SupportsMediaTypeFunc supportsMediaType;
        CanDecodeTypeFunc canDecodeType;
        CreateImageDecoderFunc createImageDecoder;
    };

    WEBCORE_EXPORT static void installFactory(ImageDecoderFactory&&);
    WEBCORE_EXPORT static void resetFactories();
    WEBCORE_EXPORT static void clearFactories();
#endif

    virtual size_t bytesDecodedToDetermineProperties() const = 0;

    virtual EncodedDataStatus encodedDataStatus() const = 0;
    virtual void setEncodedDataStatusChangeCallback(Function<void(EncodedDataStatus)>&&) { }
    virtual bool isSizeAvailable() const { return encodedDataStatus() >= EncodedDataStatus::SizeAvailable; }
    virtual IntSize size() const = 0;
    virtual size_t frameCount() const = 0;
    virtual size_t primaryFrameIndex() const { return 0; }
    virtual RepetitionCount repetitionCount() const = 0;
    virtual String uti() const { return emptyString(); }
    virtual String filenameExtension() const = 0;
    virtual String accessibilityDescription() const { return emptyString(); };
    virtual std::optional<IntPoint> hotSpot() const = 0;

#if ENABLE(QUICKLOOK_FULLSCREEN)
    virtual bool shouldUseQuickLookForFullscreen() const { return false; }
#endif

#if ENABLE(SPATIAL_IMAGE_DETECTION)
    virtual bool isSpatial() const { return false; }
#endif

    virtual IntSize frameSizeAtIndex(size_t, SubsamplingLevel = SubsamplingLevel::Default) const = 0;
    virtual bool frameIsCompleteAtIndex(size_t) const = 0;
    virtual ImageOrientation frameOrientationAtIndex(size_t) const { return ImageOrientation::Orientation::None; }
    virtual std::optional<IntSize> frameDensityCorrectedSizeAtIndex(size_t) const { return std::nullopt; }

    virtual Seconds frameDurationAtIndex(size_t) const = 0;
    virtual bool frameHasAlphaAtIndex(size_t) const = 0;
    virtual unsigned frameBytesAtIndex(size_t, SubsamplingLevel = SubsamplingLevel::Default) const = 0;

    WEBCORE_EXPORT virtual bool fetchFrameMetaDataAtIndex(size_t, SubsamplingLevel, const DecodingOptions&, ImageFrame&) const;

    virtual PlatformImagePtr createFrameImageAtIndex(size_t, SubsamplingLevel = SubsamplingLevel::Default, const DecodingOptions& = DecodingOptions(DecodingMode::Synchronous)) = 0;

    virtual void setExpectedContentSize(long long) { }
    virtual void setData(const FragmentedSharedBuffer&, bool allDataReceived) = 0;
    virtual bool isAllDataReceived() const = 0;
    virtual void clearFrameBufferCache(size_t) = 0;

protected:
    WEBCORE_EXPORT ImageDecoder();
};

} // namespace WebCore
