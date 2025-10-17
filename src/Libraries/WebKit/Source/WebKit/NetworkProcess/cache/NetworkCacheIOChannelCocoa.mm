/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 16, 2023.
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
#import "config.h"
#import "NetworkCacheIOChannel.h"

#import "Logging.h"
#import "NetworkCacheFileSystem.h"
#import <dispatch/dispatch.h>
#import <mach/vm_param.h>
#import <sys/mman.h>
#import <sys/stat.h>
#import <wtf/BlockPtr.h>
#import <wtf/text/CString.h>

namespace WebKit {
namespace NetworkCache {

static long dispatchQueueIdentifier(WorkQueue::QOS qos)
{
    switch (qos) {
    case WorkQueue::QOS::UserInteractive:
    case WorkQueue::QOS::UserInitiated:
    case WorkQueue::QOS::Default:
        return DISPATCH_QUEUE_PRIORITY_DEFAULT;
    case WorkQueue::QOS::Utility:
        return DISPATCH_QUEUE_PRIORITY_LOW;
    case WorkQueue::QOS::Background:
        return DISPATCH_QUEUE_PRIORITY_BACKGROUND;
    }
}

IOChannel::IOChannel(const String& filePath, Type type, std::optional<WorkQueue::QOS> qos)
{
    auto path = FileSystem::fileSystemRepresentation(filePath);
    int oflag;
    mode_t mode;
    WorkQueue::QOS dispatchQOS;

    switch (type) {
    case Type::Create:
        // We don't want to truncate any existing file (with O_TRUNC) as another thread might be mapping it.
        unlink(path.data());
        oflag = O_RDWR | O_CREAT | O_NONBLOCK;
        mode = S_IRUSR | S_IWUSR;
        dispatchQOS = qos.value_or(WorkQueue::QOS::Background);
        break;
    case Type::Write:
        oflag = O_WRONLY | O_NONBLOCK;
        mode = S_IRUSR | S_IWUSR;
        dispatchQOS = qos.value_or(WorkQueue::QOS::Background);
        break;
    case Type::Read:
        oflag = O_RDONLY | O_NONBLOCK;
        mode = 0;
        dispatchQOS = qos.value_or(WorkQueue::QOS::Default);
    }

    int fd = ::open(path.data(), oflag, mode);
    m_dispatchIO = adoptOSObject(dispatch_io_create(DISPATCH_IO_RANDOM, fd, dispatch_get_global_queue(dispatchQueueIdentifier(dispatchQOS), 0), [fd](int) {
        close(fd);
    }));
    ASSERT(m_dispatchIO.get());

    // This makes the channel read/write all data before invoking the handlers.
    dispatch_io_set_low_water(m_dispatchIO.get(), std::numeric_limits<size_t>::max());
}

IOChannel::~IOChannel()
{
    RELEASE_ASSERT(!m_wasDeleted.exchange(true));
}

void IOChannel::read(size_t offset, size_t size, Ref<WTF::WorkQueueBase>&& queue, Function<void(Data&&, int error)>&& completionHandler)
{
    bool didCallCompletionHandler = false;
    dispatch_io_read(m_dispatchIO.get(), offset, size, queue->dispatchQueue(), makeBlockPtr([protectedThis = Ref { *this }, queue, completionHandler = WTFMove(completionHandler), didCallCompletionHandler](bool done, dispatch_data_t fileData, int error) mutable {
        ASSERT_UNUSED(done, done || !didCallCompletionHandler);
        if (didCallCompletionHandler)
            return;

        Data data { OSObjectPtr<dispatch_data_t> { fileData } };
        completionHandler(WTFMove(data), error);
        didCallCompletionHandler = true;
    }).get());
}

void IOChannel::write(size_t offset, const Data& data, Ref<WTF::WorkQueueBase>&& queue, Function<void(int error)>&& completionHandler)
{
    auto dispatchData = data.dispatchData();
    dispatch_io_write(m_dispatchIO.get(), offset, dispatchData, queue->dispatchQueue(), makeBlockPtr([protectedThis = Ref { *this }, queue, completionHandler = WTFMove(completionHandler)](bool done, dispatch_data_t, int error) mutable {
        if (!done) {
            RELEASE_LOG_ERROR(NetworkCacheStorage, "IOChannel::write only part of data is written.");
            return;
        }

        completionHandler(error);
    }).get());
}

}
}
