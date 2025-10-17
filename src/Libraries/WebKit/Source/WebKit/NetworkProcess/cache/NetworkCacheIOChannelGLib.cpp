/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 17, 2021.
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
#include "NetworkCacheIOChannel.h"

#include "NetworkCacheFileSystem.h"
#include <wtf/MainThread.h>
#include <wtf/RunLoop.h>
#include <wtf/glib/GUniquePtr.h>
#include <wtf/glib/RunLoopSourcePriority.h>

namespace WebKit {
namespace NetworkCache {

IOChannel::IOChannel(const String& filePath, Type type, std::optional<WorkQueue::QOS> qos)
{
    auto path = FileSystem::fileSystemRepresentation(filePath);
    GRefPtr<GFile> file = adoptGRef(g_file_new_for_path(path.data()));

    Locker locker { m_lock };
    switch (type) {
    case Type::Create: {
        g_file_delete(file.get(), nullptr, nullptr);
        m_outputStream = adoptGRef(G_OUTPUT_STREAM(g_file_create(file.get(), static_cast<GFileCreateFlags>(G_FILE_CREATE_PRIVATE), nullptr, nullptr)));
#if !HAVE(STAT_BIRTHTIME)
        GUniquePtr<char> birthtimeString(g_strdup_printf("%" G_GUINT64_FORMAT, WallTime::now().secondsSinceEpoch().secondsAs<uint64_t>()));
        g_file_set_attribute_string(file.get(), "xattr::birthtime", birthtimeString.get(), G_FILE_QUERY_INFO_NONE, nullptr, nullptr);
#endif
        m_qos = qos.value_or(WorkQueue::QOS::Background);
        break;
    }
    case Type::Write: {
        auto ioStream = adoptGRef(g_file_open_readwrite(file.get(), nullptr, nullptr));
        m_outputStream = g_io_stream_get_output_stream(G_IO_STREAM(ioStream.get()));
        m_qos = qos.value_or(WorkQueue::QOS::Background);
        break;
    }
    case Type::Read:
        m_inputStream = adoptGRef(G_INPUT_STREAM(g_file_read(file.get(), nullptr, nullptr)));
        m_qos = qos.value_or(WorkQueue::QOS::Default);
        break;
    }
}

IOChannel::~IOChannel()
{
    RELEASE_ASSERT(!m_wasDeleted.exchange(true));
}

void IOChannel::read(size_t offset, size_t size, Ref<WTF::WorkQueueBase>&& queue, Function<void(Data&&, int error)>&& completionHandler)
{
    Locker locker { m_lock };
    if (!m_inputStream) {
        queue->dispatch([protectedThis = Ref { *this }, completionHandler = WTFMove(completionHandler)] {
            Data data;
            completionHandler(WTFMove(data), -1);
        });
        return;
    }

    Thread::create("IOChannel::read"_s, [this, protectedThis = Ref { *this }, offset, size, queue = Ref { queue }, completionHandler = WTFMove(completionHandler)]() mutable {
        Locker locker { m_lock };
        GRefPtr<GFileInfo> info = adoptGRef(g_file_input_stream_query_info(G_FILE_INPUT_STREAM(m_inputStream.get()), G_FILE_ATTRIBUTE_STANDARD_SIZE, nullptr, nullptr));
        if (info) {
            auto fileSize = g_file_info_get_size(info.get());
            if (fileSize && static_cast<guint64>(fileSize) <= std::numeric_limits<size_t>::max()) {
                if (G_IS_SEEKABLE(m_inputStream.get()) && g_seekable_can_seek(G_SEEKABLE(m_inputStream.get())))
                    g_seekable_seek(G_SEEKABLE(m_inputStream.get()), offset, G_SEEK_SET, nullptr, nullptr);

                size_t bufferSize = std::min<size_t>(size, fileSize - offset);
                uint8_t* bufferData = static_cast<uint8_t*>(fastMalloc(bufferSize));
                GRefPtr<GBytes> buffer = adoptGRef(g_bytes_new_with_free_func(bufferData, bufferSize, fastFree, bufferData));
                gsize bytesRead;
                if (g_input_stream_read_all(m_inputStream.get(), bufferData, bufferSize, &bytesRead, nullptr, nullptr)) {
                    GRefPtr<GBytes> bytes = bufferSize == bytesRead ? buffer : adoptGRef(g_bytes_new_from_bytes(buffer.get(), 0, bytesRead));
                    queue->dispatch([protectedThis = WTFMove(protectedThis), bytes = WTFMove(bytes), completionHandler = WTFMove(completionHandler)]() mutable {
                        Data data(WTFMove(bytes));
                        completionHandler(WTFMove(data), 0);
                    });
                    return;
                }
            }
        }
        queue->dispatch([protectedThis = WTFMove(protectedThis), completionHandler = WTFMove(completionHandler)] {
            completionHandler(Data { }, -1);
        });
    }, ThreadType::Unknown, m_qos)->detach();
}


void IOChannel::write(size_t offset, const Data& data, Ref<WTF::WorkQueueBase>&& queue, Function<void(int error)>&& completionHandler)
{
    Locker locker { m_lock };
    if (!m_outputStream) {
        queue->dispatch([protectedThis = Ref { *this }, completionHandler = WTFMove(completionHandler)] {
            completionHandler(-1);
        });
        return;
    }

    GRefPtr<GBytes> buffer = offset ? adoptGRef(g_bytes_new_from_bytes(data.bytes(), offset, data.size() - offset)) : data.bytes();
    Thread::create("IOChannel::write"_s, [this, protectedThis = Ref { *this }, buffer = WTFMove(buffer), queue = WTFMove(queue), completionHandler = WTFMove(completionHandler)]() mutable {
        Locker locker { m_lock };
        gsize buffersize;
        const auto* bufferData = g_bytes_get_data(buffer.get(), &buffersize);
        auto success = g_output_stream_write_all(m_outputStream.get(), bufferData, buffersize, nullptr, nullptr, nullptr);
        queue->dispatch([protectedThis = WTFMove(protectedThis), success, completionHandler = WTFMove(completionHandler)] {
            completionHandler(success ? 0 : -1);
        });
    }, ThreadType::Unknown, m_qos)->detach();
}

} // namespace NetworkCache
} // namespace WebKit
