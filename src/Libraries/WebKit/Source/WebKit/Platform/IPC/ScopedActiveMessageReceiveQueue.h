/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 17, 2021.
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

#include <memory>
#include <utility>
#include <wtf/HashTraits.h>
#include <wtf/Ref.h>
#include <wtf/RefPtr.h>

namespace IPC {

// Scoped holder for objects that have active message receive queues. Enforces that
// once holder stops holding the object, the message queue should be removed.
// The convention is to call stopListeningForIPC() for the object.
// Optionally the object can obtain the owning reference that was used to hold the object
// in order to attempt to avoid needless destruction of the object in holder thread.
// This useful in the case the stop invokes a cleanup task in the message handler
// thread.
template <typename T, typename HolderType = RefPtr<T>>
class ScopedActiveMessageReceiveQueue {
public:
    ScopedActiveMessageReceiveQueue() = default;
    template <typename U>
    ScopedActiveMessageReceiveQueue(U&& object)
        : m_object(WTFMove(object))
    {
    }
    ScopedActiveMessageReceiveQueue(ScopedActiveMessageReceiveQueue&& other)
        : m_object(std::exchange(other.m_object, nullptr))
    {
    }
    ScopedActiveMessageReceiveQueue& operator=(ScopedActiveMessageReceiveQueue&& other)
    {
        if (this != &other) {
            reset();
            m_object = std::exchange(other.m_object, nullptr);
        }
        return *this;
    }
    ~ScopedActiveMessageReceiveQueue()
    {
        reset();
    }
    void reset()
    {
        if (!m_object)
            return;
        stopListeningForIPCAndRelease(m_object);
    }
    T* get() const { return m_object.get(); }
    T* operator->() const { return m_object.get(); }
private:
    template<typename U>
    static auto stopListeningForIPCAndRelease(U& object) -> decltype(object->stopListeningForIPC(object.releaseNonNull()), void())
    {
        object->stopListeningForIPC(object.releaseNonNull());
    }
    template<typename U>
    static auto stopListeningForIPCAndRelease(U& object) -> decltype(object->stopListeningForIPC(WTFMove(object)), void())
    {
        object->stopListeningForIPC(WTFMove(object));
    }
    template<typename U>
    static auto stopListeningForIPCAndRelease(U& object) -> decltype(object->stopListeningForIPC(), void())
    {
        object->stopListeningForIPC();
        object = nullptr;
    }
    HolderType m_object;
};

template<typename T>
ScopedActiveMessageReceiveQueue(std::unique_ptr<T>&&) -> ScopedActiveMessageReceiveQueue<T, std::unique_ptr<T>>;

template<typename T>
ScopedActiveMessageReceiveQueue(Ref<T>&&) -> ScopedActiveMessageReceiveQueue<T, RefPtr<T>>;

}

namespace WTF {

template<typename T, typename HolderType> struct HashTraits<IPC::ScopedActiveMessageReceiveQueue<T, HolderType>> : GenericHashTraits<IPC::ScopedActiveMessageReceiveQueue<T, HolderType>> {
    using PeekType = T*;
    static T* peek(const IPC::ScopedActiveMessageReceiveQueue<T, HolderType>& value) { return value.get(); }
};

}
