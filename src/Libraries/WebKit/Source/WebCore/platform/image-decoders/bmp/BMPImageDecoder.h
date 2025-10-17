/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 3, 2023.
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

#include "BMPImageReader.h"

namespace WebCore {

// This class decodes the BMP image format.
class BMPImageDecoder final : public ScalableImageDecoder {
public:
    static Ref<ScalableImageDecoder> create(AlphaOption alphaOption, GammaAndColorProfileOption gammaAndColorProfileOption)
    {
        return adoptRef(*new BMPImageDecoder(alphaOption, gammaAndColorProfileOption));
    }

    // ScalableImageDecoder
    String filenameExtension() const final { return "bmp"_s; }
    void setData(const FragmentedSharedBuffer&, bool allDataReceived) final;
    ScalableImageDecoderFrame* frameBufferAtIndex(size_t index) final;
    // CAUTION: setFailed() deletes |m_reader|. Be careful to avoid
    // accessing deleted memory, especially when calling this from inside
    // BMPImageReader!
    bool setFailed() final;

private:
    BMPImageDecoder(AlphaOption, GammaAndColorProfileOption);
    void tryDecodeSize(bool allDataReceived) final { decode(true, allDataReceived); }

    inline uint32_t readUint32(int offset) const
    {
        return BMPImageReader::readUint32(*m_data, m_decodedOffset + offset);
    }

    // Decodes the image. If |onlySize| is true, stops decoding after
    // calculating the image size. If decoding fails but there is no more
    // data coming, sets the "decode failure" flag.
    void decode(bool onlySize, bool allDataReceived);

    // Decodes the image. If |onlySize| is true, stops decoding after
    // calculating the image size. Returns whether decoding succeeded.
    bool decodeHelper(bool onlySize);

    // Processes the file header at the beginning of the data. Sets
    // |*imgDataOffset| based on the header contents. Returns true if the
    // file header could be decoded.
    bool processFileHeader(size_t* imgDataOffset);

    // An index into |m_data| representing how much we've already decoded.
    // Note that this only tracks data _this_ class decodes; once the
    // BMPImageReader takes over this will not be updated further.
    size_t m_decodedOffset;

    // The reader used to do most of the BMP decoding.
    std::unique_ptr<BMPImageReader> m_reader;
};

} // namespace WebCore
