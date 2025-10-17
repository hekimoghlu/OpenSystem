/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 5, 2023.
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

#include <wtf/Threading.h>

namespace WTF {

template <typename IntegralType>
class SingleThreadIntegralWrapper {
public:
    SingleThreadIntegralWrapper(IntegralType);

    operator IntegralType() const;
    explicit operator bool() const;
    SingleThreadIntegralWrapper& operator=(IntegralType);
    SingleThreadIntegralWrapper& operator++();
    SingleThreadIntegralWrapper& operator--();

    IntegralType valueWithoutThreadCheck() const
    {
        // This is called after the destructor in WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR,
        // the compiler can see it as uninitialized.
        IGNORE_GCC_WARNINGS_BEGIN("uninitialized")
        return m_value;
        IGNORE_GCC_WARNINGS_END
    }

private:
#if ASSERT_ENABLED && !USE(WEB_THREAD)
    void assertThread() const { ASSERT(m_thread.ptr() == &Thread::current()); }
#else
    constexpr void assertThread() const { }
#endif

    IntegralType m_value;
#if ASSERT_ENABLED && !USE(WEB_THREAD)
    Ref<Thread> m_thread;
#endif
};

template <typename IntegralType>
inline SingleThreadIntegralWrapper<IntegralType>::SingleThreadIntegralWrapper(IntegralType value)
    : m_value { value }
#if ASSERT_ENABLED && !USE(WEB_THREAD)
    , m_thread { Thread::current() }
#endif
{ }

template <typename IntegralType>
inline SingleThreadIntegralWrapper<IntegralType>::operator IntegralType() const
{
    assertThread();
    return m_value;
}

template <typename IntegralType>
inline SingleThreadIntegralWrapper<IntegralType>::operator bool() const
{
    assertThread();
    return m_value;
}

template <typename IntegralType>
inline SingleThreadIntegralWrapper<IntegralType>& SingleThreadIntegralWrapper<IntegralType>::operator=(IntegralType value)
{
    assertThread();
    m_value = value;
    return *this;
}

template <typename IntegralType>
inline SingleThreadIntegralWrapper<IntegralType>& SingleThreadIntegralWrapper<IntegralType>::operator++()
{
    assertThread();
    m_value++;
    return *this;
}

template <typename IntegralType>
inline SingleThreadIntegralWrapper<IntegralType>& SingleThreadIntegralWrapper<IntegralType>::operator--()
{
    assertThread();
    m_value--;
    return *this;
}

} // namespace WTF

using WTF::SingleThreadIntegralWrapper;
