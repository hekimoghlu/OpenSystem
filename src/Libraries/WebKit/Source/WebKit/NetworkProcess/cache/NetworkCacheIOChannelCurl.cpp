/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 3, 2023.
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

#include <wtf/RunLoop.h>

namespace WebKit {
namespace NetworkCache {

IOChannel::IOChannel(const String& filePath, Type type, std::optional<WorkQueue::QOS>)
{
    FileSystem::FileOpenMode mode { };
    switch (type) {
    case Type::Read:
        mode = FileSystem::FileOpenMode::Read;
        break;
    case Type::Write:
        mode = FileSystem::FileOpenMode::ReadWrite;
        break;
    case Type::Create:
        mode = FileSystem::FileOpenMode::ReadWrite;
        break;
    }

    Locker locker { m_lock };
    m_fileDescriptor = FileSystem::openFile(filePath, mode);
}

IOChannel::~IOChannel()
{
    Locker locker { m_lock };
    FileSystem::closeFile(m_fileDescriptor);
}

void IOChannel::read(size_t offset, size_t size, Ref<WTF::WorkQueueBase>&& queue, Function<void(Data&&, int error)>&& completionHandler)
{
    queue->dispatch([this, protectedThis = Ref { *this }, offset, size, completionHandler = WTFMove(completionHandler)] {
        m_lock.lock();

        auto fileSize = FileSystem::fileSize(m_fileDescriptor);
        if (!fileSize || *fileSize > std::numeric_limits<size_t>::max()) {
            m_lock.unlock();
            return completionHandler(Data { }, -1);
        }

        size_t readSize = *fileSize;
        readSize = std::min(size, readSize);
        Vector<uint8_t> buffer(readSize);
        FileSystem::seekFile(m_fileDescriptor, offset, FileSystem::FileSeekOrigin::Beginning);
        int err = FileSystem::readFromFile(m_fileDescriptor, buffer.mutableSpan());
        m_lock.unlock();

        err = err < 0 ? err : 0;
        auto data = Data(WTFMove(buffer));
        completionHandler(WTFMove(data), err);
    });
}

void IOChannel::write(size_t offset, const Data& data, Ref<WTF::WorkQueueBase>&& queue, Function<void(int error)>&& completionHandler)
{
    queue->dispatch([this, protectedThis = Ref { *this }, offset, data, completionHandler = WTFMove(completionHandler)] {
        int err = 0;
        {
            Locker locker { m_lock };
            FileSystem::seekFile(m_fileDescriptor, offset, FileSystem::FileSeekOrigin::Beginning);
            err = FileSystem::writeToFile(m_fileDescriptor, data.span());
            err = err < 0 ? err : 0;
        }

        completionHandler(err);
    });
}

} // namespace NetworkCache
} // namespace WebKit
