/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 22, 2025.
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
#include "SourceBufferParser.h"

#if ENABLE(MEDIA_SOURCE)

#include "ContentType.h"
#include "SharedBuffer.h"
#include "SourceBufferParserAVFObjC.h"
#include "SourceBufferParserWebM.h"
#include <pal/spi/cocoa/MediaToolboxSPI.h>
#include <wtf/text/WTFString.h>

#include <pal/cocoa/MediaToolboxSoftLink.h>

namespace WebCore {

MediaPlayerEnums::SupportsType SourceBufferParser::isContentTypeSupported(const ContentType& type)
{
    MediaPlayerEnums::SupportsType supports = MediaPlayerEnums::SupportsType::IsNotSupported;
    supports = std::max(supports, SourceBufferParserWebM::isContentTypeSupported(type));
    supports = std::max(supports, SourceBufferParserAVFObjC::isContentTypeSupported(type));
    return supports;
}

RefPtr<SourceBufferParser> SourceBufferParser::create(const ContentType& type)
{
    if (SourceBufferParserWebM::isContentTypeSupported(type) != MediaPlayerEnums::SupportsType::IsNotSupported)
        return SourceBufferParserWebM::create();

    if (SourceBufferParserAVFObjC::isContentTypeSupported(type) != MediaPlayerEnums::SupportsType::IsNotSupported)
        return adoptRef(new SourceBufferParserAVFObjC());

    return nullptr;
}

static SourceBufferParser::CallOnClientThreadCallback callOnMainThreadCallback()
{
    return [](Function<void()>&& function) {
        callOnMainThread(WTFMove(function));
    };
}

void SourceBufferParser::setCallOnClientThreadCallback(CallOnClientThreadCallback&& callback)
{
    ASSERT(callback);
    m_callOnClientThreadCallback = WTFMove(callback);
}

SourceBufferParser::SourceBufferParser()
    : m_callOnClientThreadCallback(callOnMainThreadCallback())
{
}

void SourceBufferParser::setMinimumAudioSampleDuration(float)
{
}

SourceBufferParser::Segment::Segment(Ref<SharedBuffer>&& buffer)
    : m_segment(WTFMove(buffer))
{
}

size_t SourceBufferParser::Segment::size() const
{
    return m_segment->size();
}

auto SourceBufferParser::Segment::read(std::span<uint8_t> destination, size_t position) const -> ReadResult
{
    size_t segmentSize = size();
    destination = destination.first(std::min(destination.size(), segmentSize - std::min(position, segmentSize)));
    m_segment->copyTo(destination, position);
    return destination.size();
}

Ref<SharedBuffer> SourceBufferParser::Segment::takeSharedBuffer()
{
    return std::exchange(m_segment, SharedBuffer::create());
}

Ref<SharedBuffer> SourceBufferParser::Segment::getData(size_t offet, size_t length) const
{
    return m_segment->getContiguousData(offet, length);
}

} // namespace WebCore

#endif // ENABLE(MEDIA_SOURCE)
