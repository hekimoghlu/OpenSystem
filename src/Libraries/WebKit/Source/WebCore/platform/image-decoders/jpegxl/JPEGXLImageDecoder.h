/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 24, 2024.
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

#include "ScalableImageDecoder.h"

#if USE(JPEGXL)

#include "JxlDecoderPtr.h"

#if USE(LCMS)
#include "LCMSUniquePtr.h"
#elif USE(CG)
#include <CoreGraphics/CoreGraphics.h>
#include <wtf/RetainPtr.h>
#endif

namespace WebCore {

// This class decodes the JPEG XL image format.
class JPEGXLImageDecoder final : public ScalableImageDecoder {
public:
    static RefPtr<ScalableImageDecoder> create(AlphaOption, GammaAndColorProfileOption);

    virtual ~JPEGXLImageDecoder();

    // ScalableImageDecoder
    String filenameExtension() const override { return "jxl"_s; }
    size_t frameCount() const override WTF_REQUIRES_LOCK(m_lock);
    RepetitionCount repetitionCount() const override;
    ScalableImageDecoderFrame* frameBufferAtIndex(size_t index) override WTF_REQUIRES_LOCK(m_lock);
    void clearFrameBufferCache(size_t clearBeforeFrame) override WTF_REQUIRES_LOCK(m_lock);

    bool setFailed() override;

private:
    enum class Query {
        // This query is used for tryDecodeSize().
        Size,
        // We define a query for frame count because JPEG XL doesn't have frame count information in its code stream
        // so we need to scan the code stream to get the frame count for animated JPEG XL.
        // JPEG XL container can have frame count metadata but currently libjxl doesn't support it.
        FrameCount,
        // Query to decode a single frame.
        DecodedImage,
    };

    JPEGXLImageDecoder(AlphaOption, GammaAndColorProfileOption);

    void clear();

    void tryDecodeSize(bool allDataReceived) override WTF_REQUIRES_LOCK(m_lock);

    bool hasAlpha() const;
    bool hasAnimation() const;

    void ensureDecoderInitialized();
    bool shouldRewind(Query , size_t frameIndex) const;
    void rewind();
    void updateFrameCount() WTF_REQUIRES_LOCK(m_lock);

    void decode(Query, size_t frameIndex, bool allDataReceived) WTF_REQUIRES_LOCK(m_lock);
    JxlDecoderStatus processInput(Query) WTF_REQUIRES_LOCK(m_lock);
    static void imageOutCallback(void*, size_t x, size_t y, size_t numPixels, const void* pixels);
    void imageOut(size_t x, size_t y, size_t numPixels, const uint8_t* pixels) WTF_REQUIRES_LOCK(m_lock);

    void clearColorTransform();
    void prepareColorTransform();
    void maybePerformColorSpaceConversion(std::span<uint8_t> inputBuffer, std::span<uint8_t> outputBuffer, unsigned numberOfPixels);
#if USE(LCMS)
    LCMSProfilePtr tryDecodeICCColorProfile();
#elif USE(CG)
    RetainPtr<CGColorSpaceRef> tryDecodeICCColorProfile();
#endif

    JxlDecoderPtr m_decoder;
    size_t m_readOffset { 0 };
    std::optional<JxlBasicInfo> m_basicInfo;

    Query m_lastQuery { Query::Size };
    size_t m_frameCount { 1 };
    size_t m_currentFrame { 0 };

    bool m_isLastFrameHeaderReceived { false }; // If this is true, we know we don't need to update m_frameCount.

#if USE(LCMS)
    LCMSTransformPtr m_iccTransform;
#elif USE(CG)
    RetainPtr<CGColorSpaceRef> m_profile;
#endif
};

} // namespace WebCore

#endif // USE(JPEGXL)
