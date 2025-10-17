/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 6, 2025.
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

#include "webp/decode.h"
#include "webp/demux.h"

namespace WebCore {

class WEBPImageDecoder final : public ScalableImageDecoder {
public:
    static Ref<ScalableImageDecoder> create(AlphaOption alphaOption, GammaAndColorProfileOption gammaAndColorProfileOption)
    {
        return adoptRef(*new WEBPImageDecoder(alphaOption, gammaAndColorProfileOption));
    }

    virtual ~WEBPImageDecoder();

    String filenameExtension() const override { return "webp"_s; }
    void setData(const FragmentedSharedBuffer&, bool) final;
    ScalableImageDecoderFrame* frameBufferAtIndex(size_t index) override;
    RepetitionCount repetitionCount() const override;
    size_t frameCount() const override { return m_frameCount; }
    void clearFrameBufferCache(size_t) override;

private:
    WEBPImageDecoder(AlphaOption, GammaAndColorProfileOption);
    void tryDecodeSize(bool) override { parseHeader(); }
    void decode(size_t, bool);
    void decodeFrame(size_t, WebPDemuxer*);
    void parseHeader();
    bool initFrameBuffer(size_t, const WebPIterator*);
    void applyPostProcessing(size_t, WebPIDecoder*, WebPDecBuffer&, bool);
    size_t findFirstRequiredFrameToDecode(size_t, WebPDemuxer*);

    int m_repetitionCount { 0 };
    size_t m_frameCount { 0 };
    int m_formatFlags { 0 };
    bool m_headerParsed { false };
};

} // namespace WebCore
