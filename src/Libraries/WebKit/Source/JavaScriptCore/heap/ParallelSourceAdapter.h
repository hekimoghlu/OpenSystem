/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 9, 2024.
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

#include <wtf/Lock.h>
#include <wtf/SharedTask.h>

namespace JSC {

template<typename OuterType, typename InnerType, typename UnwrapFunc>
class ParallelSourceAdapter final : public SharedTask<InnerType()> {
public:
    ParallelSourceAdapter(RefPtr<SharedTask<OuterType()>> outerSource, const UnwrapFunc& unwrapFunc)
        : m_outerSource(outerSource)
        , m_unwrapFunc(unwrapFunc)
    {
    }
    
    InnerType run() final
    {
        Locker locker { m_lock };
        do {
            if (m_innerSource) {
                if (InnerType result = m_innerSource->run())
                    return result;
                m_innerSource = nullptr;
            }
            
            m_innerSource = m_unwrapFunc(m_outerSource->run());
        } while (m_innerSource);
        return InnerType();
    }

private:
    RefPtr<SharedTask<OuterType()>> m_outerSource;
    RefPtr<SharedTask<InnerType()>> m_innerSource;
    UnwrapFunc m_unwrapFunc;
    Lock m_lock;
};

template<typename OuterType, typename InnerType, typename UnwrapFunc>
Ref<ParallelSourceAdapter<OuterType, InnerType, UnwrapFunc>> createParallelSourceAdapter(RefPtr<SharedTask<OuterType()>> outerSource, const UnwrapFunc& unwrapFunc)
{
    return adoptRef(*new ParallelSourceAdapter<OuterType, InnerType, UnwrapFunc>(outerSource, unwrapFunc));
}
    
} // namespace JSC

