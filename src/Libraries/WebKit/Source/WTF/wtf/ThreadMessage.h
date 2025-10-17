/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 27, 2023.
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

#include <wtf/ScopedLambda.h>
#include <wtf/Threading.h>

namespace WTF {

using ThreadMessage = ScopedLambda<void(PlatformRegisters&)>;

enum class MessageStatus {
    MessageRan,
    ThreadExited,
};

// This method allows us to send a message which will be run in a signal handler on the desired thread.
// There are several caveates to this method however, This function uses signals so your message should
// be sync signal safe.
WTF_EXPORT_PRIVATE MessageStatus sendMessageScoped(const ThreadSuspendLocker&, Thread&, const ThreadMessage&);

template<typename Functor>
MessageStatus sendMessage(const ThreadSuspendLocker& locker, Thread& targetThread, const Functor& func)
{
    auto lambda = scopedLambdaRef<void(PlatformRegisters&)>(func);
    return sendMessageScoped(locker, targetThread, lambda);
}

} // namespace WTF

using WTF::sendMessage;
