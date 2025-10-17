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

#include "JSCJSValue.h"
#include "Weak.h"

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

class WeakHandleOwner;

class WeakImpl {
public:
    enum State {
        Live        = 0x0,
        Dead        = 0x1,
        Finalized   = 0x2,
        Deallocated = 0x3
    };

    enum {
        StateMask   = 0x3
    };

    WeakImpl();
    WeakImpl(JSValue, WeakHandleOwner*, void* context);

    State state();
    void setState(State);
    void clear();

    const JSValue& jsValue();
    static constexpr ptrdiff_t offsetOfJSValue() { return OBJECT_OFFSETOF(WeakImpl, m_jsValue); }
    WeakHandleOwner* weakHandleOwner();
    static constexpr ptrdiff_t offsetOfWeakHandleOwner() { return OBJECT_OFFSETOF(WeakImpl, m_weakHandleOwner); }
    void* context();

    static WeakImpl* asWeakImpl(JSValue*);

private:
    const JSValue m_jsValue;
    WeakHandleOwner* m_weakHandleOwner;
    void* m_context { nullptr };
};

inline WeakImpl::WeakImpl()
    : m_weakHandleOwner(std::bit_cast<WeakHandleOwner*>(static_cast<uintptr_t>(Deallocated)))
{
}

inline void WeakImpl::clear()
{
    ASSERT(Deallocated >= this->state());
    m_weakHandleOwner = std::bit_cast<WeakHandleOwner*>(static_cast<uintptr_t>(Deallocated));
}

inline WeakImpl::WeakImpl(JSValue jsValue, WeakHandleOwner* weakHandleOwner, void* context)
    : m_jsValue(jsValue)
    , m_weakHandleOwner(weakHandleOwner)
    , m_context(context)
{
    ASSERT(state() == Live);
    ASSERT(m_jsValue && m_jsValue.isCell());
}

inline WeakImpl::State WeakImpl::state()
{
    return static_cast<State>(reinterpret_cast<uintptr_t>(m_weakHandleOwner) & StateMask);
}

inline void WeakImpl::setState(WeakImpl::State state)
{
    ASSERT(state >= this->state());
    m_weakHandleOwner = reinterpret_cast<WeakHandleOwner*>((reinterpret_cast<uintptr_t>(m_weakHandleOwner) & ~StateMask) | state);
}

inline const JSValue& WeakImpl::jsValue()
{
    return m_jsValue;
}

inline WeakHandleOwner* WeakImpl::weakHandleOwner()
{
    return reinterpret_cast<WeakHandleOwner*>((reinterpret_cast<uintptr_t>(m_weakHandleOwner) & ~StateMask));
}

inline void* WeakImpl::context()
{
    return m_context;
}

inline WeakImpl* WeakImpl::asWeakImpl(JSValue* slot)
{
    return reinterpret_cast_ptr<WeakImpl*>(reinterpret_cast_ptr<char*>(slot) + OBJECT_OFFSETOF(WeakImpl, m_jsValue));
}

template<typename T>
inline void Weak<T>::clear()
{
    auto* pointer = impl();
    if (!pointer)
        return;
    pointer->clear();
    m_impl = nullptr;
}

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
