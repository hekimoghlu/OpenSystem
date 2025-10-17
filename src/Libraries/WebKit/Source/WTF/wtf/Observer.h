/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 13, 2024.
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

#include <wtf/Function.h>
#include <wtf/Noncopyable.h>
#include <wtf/WeakPtr.h>

namespace WTF {
template<typename> class Observer;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<typename Out, typename... In> struct IsDeprecatedWeakRefSmartPointerException<WTF::Observer<Out(In...)>> : std::true_type { };
}

namespace WTF {

template<typename> class Observer;

template <typename Out, typename... In>
class Observer<Out(In...)> : public CanMakeWeakPtr<Observer<Out(In...)>> {
    WTF_MAKE_NONCOPYABLE(Observer);
    WTF_MAKE_FAST_ALLOCATED;
public:
    Observer(Function<Out(In...)>&& callback)
        : m_callback(WTFMove(callback))
    {
        ASSERT(m_callback);
    }

    Out operator()(In... in) const
    {
        ASSERT(m_callback);
        return m_callback(std::forward<In>(in)...);
    }

private:
    Function<Out(In...)> m_callback;
};

}

using WTF::Observer;

