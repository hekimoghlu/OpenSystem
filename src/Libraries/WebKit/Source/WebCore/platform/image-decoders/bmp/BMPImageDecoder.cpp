/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 30, 2023.
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
#include "config.h"
#include "BMPImageDecoder.h"

#include "BMPImageReader.h"

namespace WebCore {

// Number of bits in .BMP used to store the file header (doesn't match
// "sizeof(BMPImageDecoder::BitmapFileHeader)" since we omit some fields and
// don't pack).
static const size_t sizeOfFileHeader = 14;

BMPImageDecoder::BMPImageDecoder(AlphaOption alphaOption, GammaAndColorProfileOption gammaAndColorProfileOption)
    : ScalableImageDecoder(alphaOption, gammaAndColorProfileOption)
    , m_decodedOffset(0)
{
}

void BMPImageDecoder::setData(const FragmentedSharedBuffer& data, bool allDataReceived)
{
    if (failed())
        return;

    ScalableImageDecoder::setData(data, allDataReceived);
    if (m_reader)
        m_reader->setData(*m_data);
}

ScalableImageDecoderFrame* BMPImageDecoder::frameBufferAtIndex(size_t index)
{
    if (index)
        return 0;

    if (m_frameBufferCache.isEmpty())
        m_frameBufferCache.grow(1);

    auto* buffer = &m_frameBufferCache.first();
    if (!buffer->isComplete())
        decode(false, isAllDataReceived());
    return buffer;
}

bool BMPImageDecoder::setFailed()
{
    m_reader = nullptr;
    return ScalableImageDecoder::setFailed();
}

void BMPImageDecoder::decode(bool onlySize, bool allDataReceived)
{
    if (failed())
        return;

    // If we couldn't decode the image but we've received all the data, decoding
    // has failed.
    if (!decodeHelper(onlySize) && allDataReceived)
        setFailed();
    // If we're done decoding the image, we don't need the BMPImageReader
    // anymore.  (If we failed, |m_reader| has already been cleared.)
    else if (!m_frameBufferCache.isEmpty() && m_frameBufferCache.first().isComplete())
        m_reader = nullptr;
}

bool BMPImageDecoder::decodeHelper(bool onlySize)
{
    size_t imgDataOffset = 0;
    if ((m_decodedOffset < sizeOfFileHeader) && !processFileHeader(&imgDataOffset))
        return false;

    if (!m_reader) {
        m_reader = makeUnique<BMPImageReader>(this, m_decodedOffset, imgDataOffset, false);
        m_reader->setData(*m_data);
    }

    if (!m_frameBufferCache.isEmpty())
        m_reader->setBuffer(&m_frameBufferCache.first());

    return m_reader->decodeBMP(onlySize);
}

bool BMPImageDecoder::processFileHeader(size_t* imgDataOffset)
{
    ASSERT(imgDataOffset);

    // Read file header.
    ASSERT(!m_decodedOffset);
    if (m_data->size() < sizeOfFileHeader)
        return false;
    auto dataSpan = m_data->span();
    const uint16_t fileType = (dataSpan[0] << 8) | dataSpan[1];
    *imgDataOffset = readUint32(10);
    m_decodedOffset = sizeOfFileHeader;

    // See if this is a bitmap filetype we understand.
    enum {
        BMAP = 0x424D,  // "BM"
        // The following additional OS/2 2.x header values (see
        // http://www.fileformat.info/format/os2bmp/egff.htm ) aren't widely
        // decoded, and are unlikely to be in much use.
        /*
        ICON = 0x4943,  // "IC"
        POINTER = 0x5054,  // "PT"
        COLORICON = 0x4349,  // "CI"
        COLORPOINTER = 0x4350,  // "CP"
        BITMAPARRAY = 0x4241,  // "BA"
        */
    };
    return (fileType == BMAP) || setFailed();
}

} // namespace WebCore
