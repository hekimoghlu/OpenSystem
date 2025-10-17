/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 20, 2025.
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
#include <wtf/Lock.h>

class GIFImageReader;

namespace WebCore {

// This class decodes the GIF image format.
class GIFImageDecoder final : public ScalableImageDecoder {
public:
    static Ref<ScalableImageDecoder> create(AlphaOption alphaOption, GammaAndColorProfileOption gammaAndColorProfileOption)
    {
        return adoptRef(*new GIFImageDecoder(alphaOption, gammaAndColorProfileOption));
    }

    virtual ~GIFImageDecoder();

    enum GIFQuery { GIFFullQuery, GIFSizeQuery, GIFFrameCountQuery };

    // ScalableImageDecoder
    String filenameExtension() const final { return "gif"_s; }
    void setData(const FragmentedSharedBuffer& data, bool allDataReceived) final;
    bool setSize(const IntSize&) final;
    size_t frameCount() const final;
    RepetitionCount repetitionCount() const final;
    ScalableImageDecoderFrame* frameBufferAtIndex(size_t index) final;
    // CAUTION: setFailed() deletes |m_reader|. Be careful to avoid
    // accessing deleted memory, especially when calling this from inside
    // GIFImageReader!
    bool setFailed() final;
    void clearFrameBufferCache(size_t clearBeforeFrame) final;

    // Callbacks from the GIF reader.
    bool haveDecodedRow(unsigned frameIndex, const Vector<unsigned char>& rowBuffer, size_t width, size_t rowNumber, unsigned repeatCount, bool writeTransparentPixels);
    bool frameComplete(unsigned frameIndex, unsigned frameDuration, ScalableImageDecoderFrame::DisposalMethod);
    void gifComplete();

private:
    GIFImageDecoder(AlphaOption, GammaAndColorProfileOption);
    void tryDecodeSize(bool allDataReceived) final { decode(0, GIFSizeQuery, allDataReceived); }
    size_t findFirstRequiredFrameToDecode(size_t);

    // If the query is GIFFullQuery, decodes the image up to (but not
    // including) |haltAtFrame|. Otherwise, decodes as much as is needed to
    // answer the query, ignoring bitmap data. If decoding fails but there
    // is no more data coming, sets the "decode failure" flag.
    void decode(unsigned haltAtFrame, GIFQuery, bool allDataReceived);

    // Called to initialize the frame buffer with the given index, based on
    // the previous frame's disposal method. Returns true on success. On
    // failure, this will mark the image as failed.
    bool initFrameBuffer(unsigned frameIndex);

    bool m_currentBufferSawAlpha;
    mutable RepetitionCount m_repetitionCount { RepetitionCountOnce };
    std::unique_ptr<GIFImageReader> m_reader;
};

} // namespace WebCore
