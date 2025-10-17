/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 9, 2024.
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
#include <wtf/RefCounted.h>
#include <wtf/ThreadSafeWeakPtr.h>
#include <wtf/ThreadSafetyAnalysis.h>

namespace WTF {

// FunctionDispatcher is an abstract representation of something that functions can be
// dispatched to. This can for example be a run loop or a work queue.

class WTF_EXPORT_PRIVATE FunctionDispatcher {
public:
    virtual ~FunctionDispatcher();

    virtual void dispatch(Function<void ()>&&) = 0;

protected:
    FunctionDispatcher();
};

class WTF_CAPABILITY("is current") WTF_EXPORT_PRIVATE SerialFunctionDispatcher : public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<SerialFunctionDispatcher>, public FunctionDispatcher {
public:
    virtual bool isCurrent() const = 0;
};

// A GuaranteedSerialFunctionDispatcher guarantees that a dispatched function will always be run.
class GuaranteedSerialFunctionDispatcher : public SerialFunctionDispatcher {
};

inline void assertIsCurrent(const SerialFunctionDispatcher& queue) WTF_ASSERTS_ACQUIRED_CAPABILITY(queue)
{
    ASSERT(queue.isCurrent());
#if !ASSERT_ENABLED
    UNUSED_PARAM(queue);
#endif
}

} // namespace WTF

using WTF::FunctionDispatcher;
using WTF::SerialFunctionDispatcher;
using WTF::GuaranteedSerialFunctionDispatcher;
