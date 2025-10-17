/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 24, 2023.
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

#if ENABLE(WEB_CODECS)

#include "BufferSource.h"
#include "ExceptionOr.h"
#include "WebCodecsEncodedAudioChunkData.h"
#include <wtf/ThreadSafeRefCounted.h>

namespace WebCore {

class WebCodecsEncodedAudioChunkStorage : public ThreadSafeRefCounted<WebCodecsEncodedAudioChunkStorage> {
public:
    static Ref<WebCodecsEncodedAudioChunkStorage> create(WebCodecsEncodedAudioChunkType type, int64_t timestamp, std::optional<uint64_t> duration, Vector<uint8_t>&& buffer) { return create(WebCodecsEncodedAudioChunkData { type, timestamp, duration, WTFMove(buffer) }); }
    static Ref<WebCodecsEncodedAudioChunkStorage> create(WebCodecsEncodedAudioChunkData&& data) { return adoptRef(* new WebCodecsEncodedAudioChunkStorage(WTFMove(data))); }

    const WebCodecsEncodedAudioChunkData& data() const { return m_data; }
    uint64_t memoryCost() const { return m_data.buffer.size(); }

private:
    explicit WebCodecsEncodedAudioChunkStorage(WebCodecsEncodedAudioChunkData&&);

    const WebCodecsEncodedAudioChunkData m_data;
};

class WebCodecsEncodedAudioChunk : public RefCounted<WebCodecsEncodedAudioChunk> {
public:
    ~WebCodecsEncodedAudioChunk() = default;

    struct Init {
        WebCodecsEncodedAudioChunkType type { WebCodecsEncodedAudioChunkType::Key };
        int64_t timestamp { 0 };
        std::optional<uint64_t> duration;
        BufferSource data;
    };

    static Ref<WebCodecsEncodedAudioChunk> create(Init&& init) { return adoptRef(*new WebCodecsEncodedAudioChunk(WTFMove(init))); }
    static Ref<WebCodecsEncodedAudioChunk> create(Ref<WebCodecsEncodedAudioChunkStorage>&& storage) { return adoptRef(*new WebCodecsEncodedAudioChunk(WTFMove(storage))); }

    WebCodecsEncodedAudioChunkType type() const { return m_storage->data().type; };
    int64_t timestamp() const { return m_storage->data().timestamp; }
    std::optional<uint64_t> duration() const { return m_storage->data().duration; }
    size_t byteLength() const { return m_storage->data().buffer.size(); }

    ExceptionOr<void> copyTo(BufferSource&&);

    std::span<const uint8_t> span() const { return m_storage->data().buffer.span(); }
    WebCodecsEncodedAudioChunkStorage& storage() { return m_storage.get(); }

private:
    explicit WebCodecsEncodedAudioChunk(Init&&);
    explicit WebCodecsEncodedAudioChunk(Ref<WebCodecsEncodedAudioChunkStorage>&&);

    Ref<WebCodecsEncodedAudioChunkStorage> m_storage;
};

inline WebCodecsEncodedAudioChunkStorage::WebCodecsEncodedAudioChunkStorage(WebCodecsEncodedAudioChunkData&& data)
    : m_data { WTFMove(data) }
{
}

inline WebCodecsEncodedAudioChunk::WebCodecsEncodedAudioChunk(Ref<WebCodecsEncodedAudioChunkStorage>&& storage)
    : m_storage(WTFMove(storage))
{
}

}

#endif // ENABLE(WEB_CODECS)
