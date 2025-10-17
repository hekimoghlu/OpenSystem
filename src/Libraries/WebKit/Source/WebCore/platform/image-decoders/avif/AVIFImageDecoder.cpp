/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 16, 2022.
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

#if USE(AVIF)

#include "AVIFImageDecoder.h"

#include "AVIFImageReader.h"

namespace WebCore {

AVIFImageDecoder::AVIFImageDecoder(AlphaOption alphaOption, GammaAndColorProfileOption gammaAndColorProfileOption)
    : ScalableImageDecoder(alphaOption, gammaAndColorProfileOption)
{
}

AVIFImageDecoder::~AVIFImageDecoder() = default;

RepetitionCount AVIFImageDecoder::repetitionCount() const
{
    if (failed() || m_frameCount <= 1)
        return RepetitionCountOnce;

    return m_repetitionCount ? m_repetitionCount : RepetitionCountInfinite;
}

size_t AVIFImageDecoder::findFirstRequiredFrameToDecode(size_t frameIndex)
{
    // The first frame doesn't depend on any other.
    if (!frameIndex)
        return 0;

    size_t firstIncompleteFrame = frameIndex;
    while (firstIncompleteFrame > 0) {
        if (m_frameBufferCache[firstIncompleteFrame - 1].isComplete())
            break;
        --firstIncompleteFrame;
    }

    return firstIncompleteFrame;
}

ScalableImageDecoderFrame* AVIFImageDecoder::frameBufferAtIndex(size_t index)
{
    const size_t imageCount = frameCount();
    if (index >= imageCount)
        return nullptr;

    if ((m_frameBufferCache.size() > index) && m_frameBufferCache[index].isComplete())
        return &m_frameBufferCache[index];

    if (imageCount && m_frameBufferCache.size() != imageCount)
        m_frameBufferCache.resize(imageCount);

    for (size_t i = findFirstRequiredFrameToDecode(index); i <= index; ++i) {
        if (m_frameBufferCache[i].isComplete())
            continue;
        decode(i, isAllDataReceived());
    }

    return &m_frameBufferCache[index];
}

bool AVIFImageDecoder::setFailed()
{
    m_reader = nullptr;
    return ScalableImageDecoder::setFailed();
}

bool AVIFImageDecoder::isComplete()
{
    if (m_frameBufferCache.isEmpty())
        return false;

    for (auto& frameBuffer : m_frameBufferCache) {
        if (!frameBuffer.isComplete())
            return false;
    }
    return true;
}

void AVIFImageDecoder::tryDecodeSize(bool allDataReceived)
{
    if (!m_reader)
        m_reader = makeUnique<AVIFImageReader>(this);

    if (!m_reader->parseHeader(*m_data, allDataReceived))
        return;

    m_frameCount = m_reader->imageCount();

    m_repetitionCount = m_frameCount > 1 ? RepetitionCountInfinite : RepetitionCountNone;
}

void AVIFImageDecoder::decode(size_t frameIndex, bool allDataReceived)
{
    if (failed())
        return;

    ASSERT(m_reader);
    m_reader->decodeFrame(frameIndex, m_frameBufferCache[frameIndex], *m_data);

    if (allDataReceived && !m_frameBufferCache.isEmpty() && isComplete())
        m_reader = nullptr;
}

}

#endif // USE(AVIF)
