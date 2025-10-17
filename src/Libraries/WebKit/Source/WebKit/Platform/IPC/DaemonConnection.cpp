/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 17, 2025.
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
#include "DaemonConnection.h"

#include "PrivateClickMeasurementConnection.h"
#include "WebPushDaemonConnection.h"

namespace WebKit {

namespace Daemon {

#if !PLATFORM(COCOA)

template<typename Traits>
void ConnectionToMachService<Traits>::initializeConnectionIfNeeded() const
{
}

template<typename Traits>
void ConnectionToMachService<Traits>::send(typename Traits::MessageType, EncodedMessage&&) const
{
}

template<typename Traits>
void ConnectionToMachService<Traits>::sendWithReply(typename Traits::MessageType, EncodedMessage&&, CompletionHandler<void(EncodedMessage&&)>&& completionHandler) const
{
    completionHandler({ });
}

template class ConnectionToMachService<PCM::ConnectionTraits>;

#if ENABLE(WEB_PUSH_NOTIFICATIONS)
template class ConnectionToMachService<WebPushD::ConnectionTraits>;
#endif

#endif // !PLATFORM(COCOA)

} // namespace Daemon

} // namespace WebKit
