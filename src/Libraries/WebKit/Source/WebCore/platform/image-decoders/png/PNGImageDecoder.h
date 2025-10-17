/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 28, 2023.
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
#include <array>
#include <png.h>

#if USE(LCMS)
#include "LCMSUniquePtr.h"
#endif

namespace WebCore {

    class PNGImageReader;

    // This class decodes the PNG image format.
    class PNGImageDecoder final : public ScalableImageDecoder {
    public:
        static Ref<ScalableImageDecoder> create(AlphaOption alphaOption, GammaAndColorProfileOption gammaAndColorProfileOption)
        {
            return adoptRef(*new PNGImageDecoder(alphaOption, gammaAndColorProfileOption));
        }

        virtual ~PNGImageDecoder();

        // ScalableImageDecoder
        String filenameExtension() const override { return "png"_s; }
        size_t frameCount() const override { return m_frameCount; }
        RepetitionCount repetitionCount() const override;
        ScalableImageDecoderFrame* frameBufferAtIndex(size_t index) override;
        // CAUTION: setFailed() deletes |m_reader|.  Be careful to avoid
        // accessing deleted memory, especially when calling this from inside
        // PNGImageReader!
        bool setFailed() override;

        // Callbacks from libpng
        void headerAvailable();
        void rowAvailable(unsigned char* rowBuffer, unsigned rowIndex, int interlacePass);
        void pngComplete();
        void readChunks(png_unknown_chunkp);
        void frameHeader();

        void init();
        void clearFrameBufferCache(size_t clearBeforeFrame) override;

        bool isComplete() const
        {
            if (m_frameBufferCache.isEmpty())
                return false;

            for (auto& imageFrame : m_frameBufferCache) {
                if (!imageFrame.isComplete())
                    return false;
            }

            return true;
        }

        bool isCompleteAtIndex(size_t index)
        {
            return (index < m_frameBufferCache.size() && m_frameBufferCache[index].isComplete());
        }

    private:
        PNGImageDecoder(AlphaOption, GammaAndColorProfileOption);
        void tryDecodeSize(bool allDataReceived) override { decode(true, 0, allDataReceived); }

        // Decodes the image.  If |onlySize| is true, stops decoding after
        // calculating the image size.  If decoding fails but there is no more
        // data coming, sets the "decode failure" flag.
        void decode(bool onlySize, unsigned haltAtFrame, bool allDataReceived);
        void initFrameBuffer(size_t frameIndex);
        void frameComplete();
        int processingStart(png_unknown_chunkp);
        int processingFinish();
        void fallbackNotAnimated();

        void clear();

        std::unique_ptr<PNGImageReader> m_reader;
        bool m_doNothingOnFailure;
        unsigned m_currentFrame;
        png_structp m_png;
        png_infop m_info;
        bool m_isAnimated;
        bool m_frameInfo;
        bool m_frameIsHidden;
        bool m_hasInfo;
        int m_gamma;
        size_t m_frameCount;
        unsigned m_playCount;
        unsigned m_totalFrames;
        unsigned m_sizePLTE;
        unsigned m_sizetRNS;
        unsigned m_sequenceNumber;
        unsigned m_width;
        unsigned m_height;
        unsigned m_xOffset;
        unsigned m_yOffset;
        unsigned m_delayNumerator;
        unsigned m_delayDenominator;
        unsigned m_dispose;
        unsigned m_blend;
        std::array<png_byte, 12 + 13> m_dataIHDR;
        std::array<png_byte, 12 + 256 * 3> m_dataPLTE;
        std::array<png_byte, 12 + 256> m_datatRNS;
#if USE(LCMS)
    LCMSTransformPtr m_iccTransform;
#endif

    };

} // namespace WebCore
