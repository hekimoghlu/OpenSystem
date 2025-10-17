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
#include "config.h"
#include "AsyncFileStream.h"

#include "FileStream.h"
#include "FileStreamClient.h"
#include <mutex>
#include <wtf/AutodrainedPool.h>
#include <wtf/Function.h>
#include <wtf/MainThread.h>
#include <wtf/MessageQueue.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/Threading.h>
#include <wtf/URL.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(AsyncFileStream);

struct AsyncFileStream::Internals {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;

    explicit Internals(FileStreamClient&);

    FileStream stream;
    FileStreamClient& client;
    std::atomic_bool destroyed { false };
};

inline AsyncFileStream::Internals::Internals(FileStreamClient& client)
    : client(client)
{
}

static void callOnFileThread(Function<void ()>&& function)
{
    ASSERT(isMainThread());
    ASSERT(function);

    static NeverDestroyed<MessageQueue<Function<void ()>>> queue;

    static std::once_flag createFileThreadOnce;
    std::call_once(createFileThreadOnce, [] {
        Thread::create("WebCore: AsyncFileStream"_s, [] {
            for (;;) {
                AutodrainedPool pool;

                auto function = queue.get().waitForMessage();

                // This can never be null because we never kill the MessageQueue.
                ASSERT(function);

                // This can bever be null because we never queue a function that is null.
                ASSERT(*function);

                (*function)();
            }
        });
    });

    queue.get().append(makeUnique<Function<void ()>>(WTFMove(function)));
}

AsyncFileStream::AsyncFileStream(FileStreamClient& client)
    : m_internals(makeUnique<Internals>(client))
{
    ASSERT(isMainThread());
}

AsyncFileStream::~AsyncFileStream()
{
    ASSERT(isMainThread());

    // Set flag to prevent client callbacks and also prevent queued operations from starting.
    m_internals->destroyed = true;

    // Call through file thread and back to main thread to make sure deletion happens
    // after all file thread functions and all main thread functions called from them.
    callOnFileThread([internals = WTFMove(m_internals)]() mutable {
        callOnMainThread([internals = WTFMove(internals)] {
        });
    });
}

void AsyncFileStream::perform(Function<Function<void(FileStreamClient&)>(FileStream&)>&& operation)
{
    auto& internals = *m_internals;
    callOnFileThread([&internals, operation = WTFMove(operation)] {
        // Don't do the operation if stop was already called on the main thread. Note that there is
        // a race here, but since skipping the operation is an optimization it's OK that we can't
        // guarantee exactly which operations are skipped. Note that this is also the only reason
        // we use an atomic_bool rather than just a bool for destroyed.
        if (internals.destroyed)
            return;
        callOnMainThread([&internals, mainThreadWork = operation(internals.stream)] {
            if (internals.destroyed)
                return;
            mainThreadWork(internals.client);
        });
    });
}

void AsyncFileStream::getSize(const String& path, std::optional<WallTime> expectedModificationTime)
{
    // FIXME: Explicit return type here and in all the other cases like this below is a workaround for a deficiency
    // in the Windows compiler at the time of this writing. Could remove it if that is resolved.
    perform([path = path.isolatedCopy(), expectedModificationTime](FileStream& stream) -> Function<void(FileStreamClient&)> {
        long long size = stream.getSize(path, expectedModificationTime);
        return [size](FileStreamClient& client) {
            client.didGetSize(size);
        };
    });
}

void AsyncFileStream::openForRead(const String& path, long long offset, long long length)
{
    // FIXME: Explicit return type here is a workaround for a deficiency in the Windows compiler at the time of this writing.
    perform([path = path.isolatedCopy(), offset, length](FileStream& stream) -> Function<void(FileStreamClient&)> {
        bool success = stream.openForRead(path, offset, length);
        return [success](FileStreamClient& client) {
            client.didOpen(success);
        };
    });
}

void AsyncFileStream::close()
{
    auto& internals = *m_internals;
    callOnFileThread([&internals] {
        internals.stream.close();
    });
}

void AsyncFileStream::read(std::span<uint8_t> buffer)
{
    perform([buffer](FileStream& stream) -> Function<void(FileStreamClient&)> {
        int bytesRead = stream.read(buffer);
        return [bytesRead](FileStreamClient& client) {
            client.didRead(bytesRead);
        };
    });
}

} // namespace WebCore
