/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 4, 2024.
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

#include "ImageDecoder.h"
#include <atomic>
#include <wtf/TZoneMalloc.h>

#if USE(CG)

namespace WebCore {

class SharedBuffer;

class ImageDecoderCG final : public ImageDecoder {
    WTF_MAKE_TZONE_ALLOCATED(ImageDecoderCG);
public:
    ImageDecoderCG(FragmentedSharedBuffer& data, AlphaOption, GammaAndColorProfileOption);

    static Ref<ImageDecoderCG> create(FragmentedSharedBuffer& data, AlphaOption alphaOption, GammaAndColorProfileOption gammaAndColorProfileOption)
    {
        return adoptRef(*new ImageDecoderCG(data, alphaOption, gammaAndColorProfileOption));
    }

    static bool supportsMediaType(MediaType type) { return type == MediaType::Image; }
    static bool canDecodeType(const String&);

    size_t bytesDecodedToDetermineProperties() const final;

    EncodedDataStatus encodedDataStatus() const final;
    IntSize size() const final { return IntSize(); }
    size_t frameCount() const final;
    size_t primaryFrameIndex() const final;
    RepetitionCount repetitionCount() const final;
    String uti() const final { return m_uti; }
    String filenameExtension() const final;
    String accessibilityDescription() const final;
    std::optional<IntPoint> hotSpot() const final;

    IntSize frameSizeAtIndex(size_t, SubsamplingLevel = SubsamplingLevel::Default) const final;
    bool frameIsCompleteAtIndex(size_t) const final;
    ImageOrientation frameOrientationAtIndex(size_t) const final;
    std::optional<IntSize> frameDensityCorrectedSizeAtIndex(size_t) const final;

    Seconds frameDurationAtIndex(size_t) const final;
    bool frameHasAlphaAtIndex(size_t) const final;
    unsigned frameBytesAtIndex(size_t, SubsamplingLevel = SubsamplingLevel::Default) const final;

    bool fetchFrameMetaDataAtIndex(size_t, SubsamplingLevel, const DecodingOptions&, ImageFrame&) const final;

    PlatformImagePtr createFrameImageAtIndex(size_t, SubsamplingLevel = SubsamplingLevel::Default, const DecodingOptions& = DecodingOptions(DecodingMode::Synchronous)) final;

    void setData(const FragmentedSharedBuffer&, bool allDataReceived) final;
    bool isAllDataReceived() const final { return m_isAllDataReceived; }
    void clearFrameBufferCache(size_t) final { }

    static String decodeUTI(CGImageSourceRef, const SharedBuffer&);

private:
    bool hasAlpha() const;
    String decodeUTI(const SharedBuffer&) const;

#if ENABLE(QUICKLOOK_FULLSCREEN)
    bool isSpatial() const;
    bool isMaybePanoramic() const;
    bool shouldUseQuickLookForFullscreen() const;
#endif

    bool m_isAllDataReceived { false };
    std::atomic<bool> m_isXBitmapImage { false };
    mutable EncodedDataStatus m_encodedDataStatus { EncodedDataStatus::Unknown };
    String m_uti;
    RetainPtr<CGImageSourceRef> m_nativeDecoder;
};

} // namespace WebCore

#endif // USE(CG)
