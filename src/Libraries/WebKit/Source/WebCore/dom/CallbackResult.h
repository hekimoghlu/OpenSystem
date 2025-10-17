/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 1, 2024.
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

#include <wtf/Expected.h>

namespace WebCore {

enum class CallbackResultType {
    Success,
    ExceptionThrown,
    UnableToExecute
};

template<typename ReturnType> class CallbackResult {
public:
    CallbackResult(CallbackResultType);
    CallbackResult(ReturnType&&);

    CallbackResultType type() const;
    ReturnType&& releaseReturnValue();

private:
    Expected<ReturnType, CallbackResultType> m_value;
};

template<> class CallbackResult<void> {
public:
    CallbackResult() = default;
    CallbackResult(CallbackResultType);

    CallbackResultType type() const;

private:
    CallbackResultType m_type = CallbackResultType::Success;
};


template<typename ReturnType> inline CallbackResult<ReturnType>::CallbackResult(CallbackResultType type)
    : m_value(makeUnexpected(type))
{
}

template<typename ReturnType> inline CallbackResult<ReturnType>::CallbackResult(ReturnType&& returnValue)
    : m_value(WTFMove(returnValue))
{
}

template<typename ReturnType> inline CallbackResultType CallbackResult<ReturnType>::type() const
{
    return m_value.has_value() ? CallbackResultType::Success : m_value.error();
}

template<typename ReturnType> inline auto CallbackResult<ReturnType>::releaseReturnValue() -> ReturnType&&
{
    ASSERT(m_value.has_value());
    return WTFMove(m_value.value());
}


// Void specialization

inline CallbackResult<void>::CallbackResult(CallbackResultType type)
    : m_type(type)
{
}

inline CallbackResultType CallbackResult<void>::type() const
{
    return m_type;
}

}
