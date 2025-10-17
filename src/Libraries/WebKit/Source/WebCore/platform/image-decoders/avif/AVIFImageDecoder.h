/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 15, 2021.
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

namespace WebCore {

class AVIFImageReader;

// This class decodes the AVIF image format.
class AVIFImageDecoder final : public ScalableImageDecoder {
public:
    static Ref<ScalableImageDecoder> create(AlphaOption alphaOption, GammaAndColorProfileOption gammaAndColorProfileOption)
    {
        return adoptRef(*new AVIFImageDecoder(alphaOption, gammaAndColorProfileOption));
    }

    virtual ~AVIFImageDecoder();

    // ScalableImageDecoder
    String filenameExtension() const final { return "avif"_s; }
    size_t frameCount() const final { return m_frameCount; };
    RepetitionCount repetitionCount() const final;
    ScalableImageDecoderFrame* frameBufferAtIndex(size_t index) final WTF_REQUIRES_LOCK(m_lock);
    bool setFailed() final;

private:
    AVIFImageDecoder(AlphaOption, GammaAndColorProfileOption);

    void tryDecodeSize(bool allDataReceived) final;
    void decode(size_t frameIndex, bool allDataReceived) WTF_REQUIRES_LOCK(m_lock);
    bool isComplete() WTF_REQUIRES_LOCK(m_lock);
    size_t findFirstRequiredFrameToDecode(size_t frameIndex) WTF_REQUIRES_LOCK(m_lock);

    std::unique_ptr<AVIFImageReader> m_reader { nullptr };

    size_t m_frameCount { 0 };
    int m_repetitionCount { 0 };
};

} // namespace WebCore
