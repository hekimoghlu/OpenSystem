/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 10, 2024.
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
#include <wtf/WorkQueue.h>

#include <wtf/BlockPtr.h>
#include <wtf/Ref.h>

namespace WTF {

namespace {

struct DispatchWorkItem {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;
    Ref<WorkQueueBase> m_workQueue;
    Function<void()> m_function;
    void operator()() { m_function(); }
};

}

template<typename T> static void dispatchWorkItem(void* dispatchContext)
{
    T* item = reinterpret_cast<T*>(dispatchContext);
    (*item)();
    delete item;
}

void WorkQueueBase::dispatch(Function<void()>&& function)
{
    dispatch_async_f(m_dispatchQueue.get(), new DispatchWorkItem { Ref { *this }, WTFMove(function) }, dispatchWorkItem<DispatchWorkItem>);
}

void WorkQueueBase::dispatchWithQOS(Function<void()>&& function, QOS qos)
{
    dispatch_block_t blockWithQOS = dispatch_block_create_with_qos_class(DISPATCH_BLOCK_ENFORCE_QOS_CLASS, Thread::dispatchQOSClass(qos), 0, makeBlockPtr([function = WTFMove(function)] () mutable {
        function();
        function = { };
    }).get());
    dispatch_async(m_dispatchQueue.get(), blockWithQOS);
#if !__has_feature(objc_arc)
    Block_release(blockWithQOS);
#endif
}

void WorkQueueBase::dispatchAfter(Seconds duration, Function<void()>&& function)
{
    dispatch_after_f(dispatch_time(DISPATCH_TIME_NOW, duration.nanosecondsAs<int64_t>()), m_dispatchQueue.get(), new DispatchWorkItem { Ref { *this },  WTFMove(function) }, dispatchWorkItem<DispatchWorkItem>);
}

void WorkQueueBase::dispatchSync(Function<void()>&& function)
{
    dispatch_sync_f(m_dispatchQueue.get(), new Function<void()> { WTFMove(function) }, dispatchWorkItem<Function<void()>>);
}

WorkQueueBase::WorkQueueBase(OSObjectPtr<dispatch_queue_t>&& dispatchQueue)
    : m_dispatchQueue(WTFMove(dispatchQueue))
    , m_threadID(mainThreadID)
{
}

void WorkQueueBase::platformInitialize(ASCIILiteral name, Type type, QOS qos)
{
    dispatch_queue_attr_t attr = type == Type::Concurrent ? DISPATCH_QUEUE_CONCURRENT : DISPATCH_QUEUE_SERIAL;
    attr = dispatch_queue_attr_make_with_qos_class(attr, Thread::dispatchQOSClass(qos), 0);
    m_dispatchQueue = adoptOSObject(dispatch_queue_create(name, attr));
    dispatch_set_context(m_dispatchQueue.get(), this);
    // We use &s_uid for the key, since it's convenient. Dispatch does not dereference it.
    // We use s_uid to generate the id so that WorkQueues and Threads share the id namespace.
    // This makes it possible to assert that code runs in the expected sequence, regardless of if it is
    // in a thread or a work queue.
    m_threadID = ++s_uid;
    dispatch_queue_set_specific(m_dispatchQueue.get(), &s_uid, reinterpret_cast<void*>(m_threadID), nullptr);
}

void WorkQueueBase::platformInvalidate()
{
}

WorkQueue::WorkQueue(MainTag)
    : WorkQueueBase(dispatch_get_main_queue())
{
}

void ConcurrentWorkQueue::apply(size_t iterations, WTF::Function<void(size_t index)>&& function)
{
    dispatch_apply(iterations, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), makeBlockPtr([function = WTFMove(function)](size_t index) {
        function(index);
    }).get());
}

}
