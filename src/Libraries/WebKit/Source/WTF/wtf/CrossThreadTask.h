/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 7, 2023.
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

#include <wtf/CrossThreadCopier.h>
#include <wtf/Function.h>
#include <wtf/RefPtr.h>
#include <wtf/StdLibExtras.h>
#include <wtf/ThreadSafeRefCounted.h>

namespace WTF {

class CrossThreadTask {
    WTF_MAKE_FAST_ALLOCATED;
public:
    CrossThreadTask() = default;

    CrossThreadTask(Function<void ()>&& taskFunction)
        : m_taskFunction(WTFMove(taskFunction))
    {
        ASSERT(m_taskFunction);
    }

    void performTask()
    {
        m_taskFunction();
    }

    explicit operator bool() const { return !!m_taskFunction; }

protected:
    Function<void ()> m_taskFunction;
};

template <typename F, typename ArgsTuple, size_t... ArgsIndex>
void callFunctionForCrossThreadTaskImpl(F function, ArgsTuple&& args, std::index_sequence<ArgsIndex...>)
{
    function(std::get<ArgsIndex>(std::forward<ArgsTuple>(args))...);
}

template <typename F, typename ArgsTuple, typename ArgsIndices = std::make_index_sequence<std::tuple_size<ArgsTuple>::value>>
void callFunctionForCrossThreadTask(F function, ArgsTuple&& args)
{
    callFunctionForCrossThreadTaskImpl(function, std::forward<ArgsTuple>(args), ArgsIndices());
}

template<typename... Parameters, typename... Arguments>
CrossThreadTask createCrossThreadTask(void (*method)(Parameters...), const Arguments&... arguments)
{
    return CrossThreadTask([method, arguments = std::make_tuple(crossThreadCopy(arguments)...)]() mutable {
        callFunctionForCrossThreadTask(method, WTFMove(arguments));
    });
}

template <typename C, typename MF, typename ArgsTuple, size_t... ArgsIndex>
void callMemberFunctionForCrossThreadTaskImpl(C* object, MF function, ArgsTuple&& args, std::index_sequence<ArgsIndex...>)
{
    (object->*function)(std::get<ArgsIndex>(std::forward<ArgsTuple>(args))...);
}

template <typename C, typename MF, typename ArgsTuple, typename ArgsIndices = std::make_index_sequence<std::tuple_size<ArgsTuple>::value>>
void callMemberFunctionForCrossThreadTask(C* object, MF function, ArgsTuple&& args)
{
    callMemberFunctionForCrossThreadTaskImpl(object, function, std::forward<ArgsTuple>(args), ArgsIndices());
}

template<typename T, typename... Parameters, typename... Arguments>
requires (WTF::HasRefPtrMemberFunctions<T>::value)
CrossThreadTask createCrossThreadTask(T& callee, void (T::*method)(Parameters...), const Arguments&... arguments)
{
    return CrossThreadTask([callee = RefPtr { &callee }, method, arguments = std::make_tuple(crossThreadCopy(arguments)...)]() mutable {
        callMemberFunctionForCrossThreadTask(callee.get(), method, WTFMove(arguments));
    });
}

template<typename T, typename... Parameters, typename... Arguments>
requires (!WTF::HasRefPtrMemberFunctions<T>::value)
CrossThreadTask createCrossThreadTask(T& callee, void (T::*method)(Parameters...), const Arguments&... arguments)
{
    return CrossThreadTask([callee = &callee, method, arguments = std::make_tuple(crossThreadCopy(arguments)...)]() mutable {
        callMemberFunctionForCrossThreadTask(callee, method, WTFMove(arguments));
    });
}

} // namespace WTF

using WTF::CrossThreadTask;
using WTF::createCrossThreadTask;
