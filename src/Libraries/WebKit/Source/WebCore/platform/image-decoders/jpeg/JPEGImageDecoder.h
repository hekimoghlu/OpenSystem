/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 10, 2025.
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
#include <stdio.h> // Needed by jpeglib.h for FILE.

// ICU defines TRUE and FALSE macros, breaking libjpeg v9 headers
#undef TRUE
#undef FALSE

#if USE(LCMS)
#include "LCMSUniquePtr.h"
#endif

extern "C" {
#include <jpeglib.h>
}

namespace WebCore {

    class JPEGImageReader;

    // This class decodes the JPEG image format.
    class JPEGImageDecoder final : public ScalableImageDecoder {
    public:
        static Ref<ScalableImageDecoder> create(AlphaOption alphaOption, GammaAndColorProfileOption gammaAndColorProfileOption)
        {
            return adoptRef(*new JPEGImageDecoder(alphaOption, gammaAndColorProfileOption));
        }

        virtual ~JPEGImageDecoder();

        // ScalableImageDecoder
        String filenameExtension() const override { return "jpg"_s; }
        ScalableImageDecoderFrame* frameBufferAtIndex(size_t index) override;
        // CAUTION: setFailed() deletes |m_reader|.  Be careful to avoid
        // accessing deleted memory, especially when calling this from inside
        // JPEGImageReader!
        bool setFailed() override;

        bool outputScanlines();
        void jpegComplete();

        void setOrientation(ImageOrientation orientation) { m_orientation = orientation; }
#if USE(LCMS)
        void setICCProfile(RefPtr<SharedBuffer>&&);
#endif

    private:
        JPEGImageDecoder(AlphaOption, GammaAndColorProfileOption);
        void tryDecodeSize(bool allDataReceived) override { decode(true, allDataReceived); }

        // Decodes the image.  If |onlySize| is true, stops decoding after
        // calculating the image size.  If decoding fails but there is no more
        // data coming, sets the "decode failure" flag.
        void decode(bool onlySize, bool allDataReceived);

        template <J_COLOR_SPACE colorSpace>
        bool outputScanlines(ScalableImageDecoderFrame& buffer);

        template <J_COLOR_SPACE colorSpace, bool isScaled>
        bool outputScanlines(ScalableImageDecoderFrame& buffer);

        void clear();

        std::unique_ptr<JPEGImageReader> m_reader;
#if USE(LCMS)
        LCMSTransformPtr m_iccTransform;
#endif
    };

} // namespace WebCore
