/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 17, 2024.
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
#ifndef StorageThread_h
#define StorageThread_h

#include <wtf/CheckedRef.h>
#include <wtf/Function.h>
#include <wtf/MessageQueue.h>
#include <wtf/Threading.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class StorageAreaSync;
class StorageTask;

class StorageThread final : public CanMakeCheckedPtr<StorageThread> {
    WTF_MAKE_NONCOPYABLE(StorageThread);
    WTF_MAKE_FAST_ALLOCATED;
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(StorageThread);
public:
    enum class Type { LocalStorage, IndexedDB };
    StorageThread(Type = Type::LocalStorage);
    ~StorageThread();

    void start();
    void terminate();

    void dispatch(Function<void ()>&&);

    static void releaseFastMallocFreeMemoryInAllThreads();

private:
    void threadEntryPoint();

    // Background thread part of the terminate procedure.
    void performTerminate();

    RefPtr<Thread> m_thread;
    Type m_type;
    MessageQueue<Function<void ()>> m_queue;
};

} // namespace WebCore

#endif // StorageThread_h
