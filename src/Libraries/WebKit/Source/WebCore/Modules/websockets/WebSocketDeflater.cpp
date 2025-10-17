/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 13, 2024.
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
#include "WebSocketDeflater.h"

#include "Logging.h"
#include <array>
#include <wtf/CheckedArithmetic.h>
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMallocInlines.h>
#include <zlib.h>

namespace WebCore {

static const int defaultMemLevel = 8;
static const size_t bufferIncrementUnit = 4096;

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebSocketDeflater);

WebSocketDeflater::WebSocketDeflater(int windowBits, ContextTakeOverMode contextTakeOverMode)
    : m_windowBits(windowBits)
    , m_contextTakeOverMode(contextTakeOverMode)
{
    ASSERT(m_windowBits >= 8);
    ASSERT(m_windowBits <= 15);
    m_stream = makeUniqueWithoutFastMallocCheck<z_stream>();
    zeroBytes(*m_stream);
}

bool WebSocketDeflater::initialize()
{
    return deflateInit2(m_stream.get(), Z_DEFAULT_COMPRESSION, Z_DEFLATED, -m_windowBits, defaultMemLevel, Z_DEFAULT_STRATEGY) == Z_OK;
}

WebSocketDeflater::~WebSocketDeflater()
{
    int result = deflateEnd(m_stream.get());
    if (result != Z_OK)
        LOG(Network, "WebSocketDeflater %p Destructor deflateEnd() failed: %d is returned", this, result);
}

static void setStreamParameter(z_stream* stream, std::span<const uint8_t> inputData, std::span<uint8_t> outputData)
{
    stream->next_in = const_cast<uint8_t*>(inputData.data());
    stream->avail_in = inputData.size();
    stream->next_out = outputData.data();
    stream->avail_out = outputData.size();
}

bool WebSocketDeflater::addBytes(std::span<const uint8_t> data)
{
    if (!data.size())
        return false;

    size_t maxLength = deflateBound(m_stream.get(), data.size());
    size_t writePosition = m_buffer.size();
    CheckedSize bufferSize = maxLength;
    bufferSize += writePosition; 
    if (bufferSize.hasOverflowed())
        return false;

    m_buffer.grow(bufferSize.value());
    setStreamParameter(m_stream.get(), data, m_buffer.mutableSpan().subspan(writePosition, maxLength));
    int result = deflate(m_stream.get(), Z_NO_FLUSH);
    if (result != Z_OK || m_stream->avail_in > 0)
        return false;

    m_buffer.shrink(bufferSize.value() - m_stream->avail_out);
    return true;
}

bool WebSocketDeflater::finish()
{
    while (true) {
        size_t writePosition = m_buffer.size();
        CheckedSize bufferSize = writePosition;
        bufferSize += bufferIncrementUnit; 
        if (bufferSize.hasOverflowed())
            return false;

        m_buffer.grow(bufferSize.value());
        size_t availableCapacity = m_buffer.size() - writePosition;
        setStreamParameter(m_stream.get(), { }, m_buffer.mutableSpan().subspan(writePosition, availableCapacity));
        int result = deflate(m_stream.get(), Z_SYNC_FLUSH);
        if (m_stream->avail_out) {
            m_buffer.shrink(writePosition + availableCapacity - m_stream->avail_out);
            if (result == Z_OK)
                break;
            if (result != Z_BUF_ERROR)
                return false;
        }
    }
    // Remove 4 octets from the tail as the specification requires.
    if (m_buffer.size() <= 4)
        return false;
    m_buffer.shrink(m_buffer.size() - 4);
    return true;
}

void WebSocketDeflater::reset()
{
    m_buffer.clear();
    if (m_contextTakeOverMode == DoNotTakeOverContext)
        deflateReset(m_stream.get());
}

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebSocketInflater);

WebSocketInflater::WebSocketInflater(int windowBits)
    : m_windowBits(windowBits)
    , m_stream(makeUniqueWithoutFastMallocCheck<z_stream>())
{
    zeroBytes(*m_stream);
}

bool WebSocketInflater::initialize()
{
    return inflateInit2(m_stream.get(), -m_windowBits) == Z_OK;
}

WebSocketInflater::~WebSocketInflater()
{
    int result = inflateEnd(m_stream.get());
    if (result != Z_OK)
        LOG(Network, "WebSocketInflater %p Destructor inflateEnd() failed: %d is returned", this, result);
}

bool WebSocketInflater::addBytes(std::span<const uint8_t> data)
{
    if (!data.size())
        return false;

    size_t consumedSoFar = 0;
    while (consumedSoFar < data.size()) {
        size_t writePosition = m_buffer.size();
        CheckedSize bufferSize = writePosition;
        bufferSize += bufferIncrementUnit; 
        if (bufferSize.hasOverflowed())
            return false;

        m_buffer.grow(bufferSize.value());
        size_t availableCapacity = m_buffer.size() - writePosition;
        size_t remainingLength = data.size() - consumedSoFar;
        setStreamParameter(m_stream.get(), data.subspan(consumedSoFar, remainingLength), m_buffer.mutableSpan().subspan(writePosition, availableCapacity));
        int result = inflate(m_stream.get(), Z_NO_FLUSH);
        consumedSoFar += remainingLength - m_stream->avail_in;
        m_buffer.shrink(writePosition + availableCapacity - m_stream->avail_out);
        if (result == Z_BUF_ERROR)
            continue;
        if (result == Z_STREAM_END) {
            // Received a block with BFINAL set to 1. Reset decompression state.
            if (inflateReset(m_stream.get()) != Z_OK)
                return false;
            continue;
        }
        if (result != Z_OK)
            return false;
        ASSERT(remainingLength > m_stream->avail_in);
    }
    ASSERT(consumedSoFar == data.size());
    return true;
}

bool WebSocketInflater::finish()
{
    constexpr std::array<uint8_t, 4> strippedFields { 0, 0, 0xFF, 0xFF };

    // Appends 4 octests of 0x00 0x00 0xff 0xff
    size_t consumedSoFar = 0;
    while (consumedSoFar < strippedFields.size()) {
        size_t writePosition = m_buffer.size();
        CheckedSize bufferSize = writePosition;
        bufferSize += bufferIncrementUnit; 
        if (bufferSize.hasOverflowed())
            return false;

        m_buffer.grow(bufferSize.value());
        size_t availableCapacity = m_buffer.size() - writePosition;
        size_t remainingLength = strippedFields.size() - consumedSoFar;
        setStreamParameter(m_stream.get(), std::span { strippedFields }.subspan(consumedSoFar), m_buffer.mutableSpan().subspan(writePosition, availableCapacity));
        int result = inflate(m_stream.get(), Z_FINISH);
        consumedSoFar += remainingLength - m_stream->avail_in;
        m_buffer.shrink(writePosition + availableCapacity - m_stream->avail_out);
        if (result == Z_BUF_ERROR)
            continue;
        if (result != Z_OK && result != Z_STREAM_END)
            return false;
        ASSERT(remainingLength > m_stream->avail_in);
    }
    ASSERT(consumedSoFar == strippedFields.size());

    return true;
}

void WebSocketInflater::reset()
{
    m_buffer.clear();
}

} // namespace WebCore
