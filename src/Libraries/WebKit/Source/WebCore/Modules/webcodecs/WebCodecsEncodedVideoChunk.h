/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 13, 2022.
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
#include "WebCodecsEncodedVideoChunkData.h"
#include <wtf/ThreadSafeRefCounted.h>

namespace WebCore {

class WebCodecsEncodedVideoChunkStorage : public ThreadSafeRefCounted<WebCodecsEncodedVideoChunkStorage> {
public:
    static Ref<WebCodecsEncodedVideoChunkStorage> create(WebCodecsEncodedVideoChunkType type, int64_t timestamp, std::optional<uint64_t> duration, Vector<uint8_t>&& buffer) { return create(WebCodecsEncodedVideoChunkData { type, timestamp, duration, WTFMove(buffer) }); }
    static Ref<WebCodecsEncodedVideoChunkStorage> create(WebCodecsEncodedVideoChunkData&& data) { return adoptRef(* new WebCodecsEncodedVideoChunkStorage(WTFMove(data))); }

    const WebCodecsEncodedVideoChunkData& data() const { return m_data; }
    uint64_t memoryCost() const { return m_data.buffer.size(); }

private:
    explicit WebCodecsEncodedVideoChunkStorage(WebCodecsEncodedVideoChunkData&&);

    const WebCodecsEncodedVideoChunkData m_data;
};

class WebCodecsEncodedVideoChunk : public RefCounted<WebCodecsEncodedVideoChunk> {
public:
    ~WebCodecsEncodedVideoChunk() = default;

    struct Init {
        WebCodecsEncodedVideoChunkType type { WebCodecsEncodedVideoChunkType::Key };
        int64_t timestamp { 0 };
        std::optional<uint64_t> duration;
        BufferSource data;
    };

    static Ref<WebCodecsEncodedVideoChunk> create(Init&& init) { return adoptRef(*new WebCodecsEncodedVideoChunk(WTFMove(init))); }
    static Ref<WebCodecsEncodedVideoChunk> create(Ref<WebCodecsEncodedVideoChunkStorage>&& storage) { return adoptRef(*new WebCodecsEncodedVideoChunk(WTFMove(storage))); }

    WebCodecsEncodedVideoChunkType type() const { return m_storage->data().type; };
    int64_t timestamp() const { return m_storage->data().timestamp; }
    std::optional<uint64_t> duration() const { return m_storage->data().duration; }
    size_t byteLength() const { return m_storage->data().buffer.size(); }

    ExceptionOr<void> copyTo(BufferSource&&);

    std::span<const uint8_t> span() const { return m_storage->data().buffer.span(); }
    WebCodecsEncodedVideoChunkStorage& storage() { return m_storage.get(); }

private:
    explicit WebCodecsEncodedVideoChunk(Init&&);
    explicit WebCodecsEncodedVideoChunk(Ref<WebCodecsEncodedVideoChunkStorage>&&);

    Ref<WebCodecsEncodedVideoChunkStorage> m_storage;
};

inline WebCodecsEncodedVideoChunkStorage::WebCodecsEncodedVideoChunkStorage(WebCodecsEncodedVideoChunkData&& data)
    : m_data { WTFMove(data) }
{
}

inline WebCodecsEncodedVideoChunk::WebCodecsEncodedVideoChunk(Ref<WebCodecsEncodedVideoChunkStorage>&& storage)
    : m_storage(WTFMove(storage))
{
}

}

#endif // ENABLE(WEB_CODECS)
