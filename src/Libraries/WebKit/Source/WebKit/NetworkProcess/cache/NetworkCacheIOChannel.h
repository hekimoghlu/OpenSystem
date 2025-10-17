/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 11, 2024.
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

#include "NetworkCacheData.h"
#include <wtf/Function.h>
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/WorkQueue.h>
#include <wtf/text/WTFString.h>

#if USE(GLIB)
#include <wtf/glib/GRefPtr.h>

typedef struct _GInputStream GInputStream;
typedef struct _GOutputStream GOutputStream;
#endif

namespace WebKit {
namespace NetworkCache {

class IOChannel : public ThreadSafeRefCounted<IOChannel> {
public:
    enum class Type { Read, Write, Create };
    static Ref<IOChannel> open(const String& filePath, Type type, std::optional<WorkQueue::QOS> qos = { }) { return adoptRef(*new IOChannel(filePath, type, qos)); }

    void read(size_t offset, size_t, Ref<WTF::WorkQueueBase>&&, Function<void(Data&&, int error)>&&);
    void write(size_t offset, const Data&, Ref<WTF::WorkQueueBase>&&, Function<void(int error)>&&);

    ~IOChannel();

private:
    IOChannel(const String& filePath, IOChannel::Type, std::optional<WorkQueue::QOS>);

#if !PLATFORM(COCOA)
    Lock m_lock;
#endif
    std::atomic<bool> m_wasDeleted { false }; // Try to narrow down a crash, https://bugs.webkit.org/show_bug.cgi?id=165659
#if PLATFORM(COCOA)
    OSObjectPtr<dispatch_io_t> m_dispatchIO;
#elif USE(GLIB)
    GRefPtr<GInputStream> m_inputStream WTF_GUARDED_BY_LOCK(m_lock);
    GRefPtr<GOutputStream> m_outputStream WTF_GUARDED_BY_LOCK(m_lock);
    WorkQueue::QOS m_qos WTF_GUARDED_BY_LOCK(m_lock);
#else // !PLATFORM(COCOA) && !USE(GLIB)
    FileSystem::PlatformFileHandle m_fileDescriptor WTF_GUARDED_BY_LOCK(m_lock) { FileSystem::invalidPlatformFileHandle };
#endif
};

} // namespace NetworkCache
} // namespace WebKit
