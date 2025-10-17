/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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

#include "PlatformAudioData.h"
#include "SpanCoreAudio.h"
#include <CoreAudio/CoreAudioTypes.h>
#include <wtf/IteratorRange.h>
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

struct AudioBuffer;
struct AudioBufferList;
typedef struct OpaqueCMBlockBuffer* CMBlockBufferRef;
typedef struct opaqueCMSampleBuffer* CMSampleBufferRef;

namespace WebCore {

class CAAudioStreamDescription;

class WebAudioBufferList final : public PlatformAudioData {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(WebAudioBufferList, WEBCORE_EXPORT);
public:
    WEBCORE_EXPORT WebAudioBufferList(const CAAudioStreamDescription&);
    WEBCORE_EXPORT WebAudioBufferList(const CAAudioStreamDescription&, size_t sampleCount);
    WebAudioBufferList(const CAAudioStreamDescription&, CMSampleBufferRef);
    WEBCORE_EXPORT virtual ~WebAudioBufferList();

    static std::optional<std::pair<UniqueRef<WebAudioBufferList>, RetainPtr<CMBlockBufferRef>>> createWebAudioBufferListWithBlockBuffer(const CAAudioStreamDescription&, size_t sampleCount);

    void reset();
    WEBCORE_EXPORT void setSampleCount(size_t);

    AudioBufferList* list() const { return m_list.get(); }
    operator AudioBufferList&() const { return *m_list; }

    uint32_t bufferCount() const;
    uint32_t channelCount() const { return m_channelCount; }
    AudioBuffer* buffer(uint32_t index) const;

    template <typename T = uint8_t>
    std::span<T> bufferAsSpan(uint32_t index) const
    {
        auto buffers = span(*m_list);
        ASSERT(index < buffers.size());
        if (index < buffers.size())
            return mutableSpan<T>(buffers[index]);
        return { };
    }

    IteratorRange<AudioBuffer*> buffers() const;

    WEBCORE_EXPORT static bool isSupportedDescription(const CAAudioStreamDescription&, size_t sampleCount);

    WEBCORE_EXPORT void zeroFlatBuffer();

private:
    Kind kind() const { return Kind::WebAudioBufferList; }
    void initializeList(std::span<uint8_t>, size_t);
    RetainPtr<CMBlockBufferRef> setSampleCountWithBlockBuffer(size_t);

    size_t m_listBufferSize { 0 };
    uint32_t m_bytesPerFrame { 0 };
    uint32_t m_channelCount { 0 };
    size_t m_sampleCount { 0 };
    std::unique_ptr<AudioBufferList> m_canonicalList;
    std::unique_ptr<AudioBufferList> m_list;
    RetainPtr<CMBlockBufferRef> m_blockBuffer;
    Vector<uint8_t> m_flatBuffer;
};

}

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::WebAudioBufferList)
static bool isType(const WebCore::PlatformAudioData& data) { return data.kind() == WebCore::PlatformAudioData::Kind::WebAudioBufferList; }
SPECIALIZE_TYPE_TRAITS_END()
