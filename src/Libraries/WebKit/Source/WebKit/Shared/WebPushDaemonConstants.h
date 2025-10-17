/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 31, 2023.
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

#ifdef __cplusplus

#include <wtf/Seconds.h>
#include <wtf/text/ASCIILiteral.h>

namespace WebKit::WebPushD {

// If an origin processes more than this many silent pushes, then it will be unsubscribed from push.
constexpr unsigned maxSilentPushCount = 3;

// getPendingPushMessage starts a timer with this time interval after returning a push message to the client. If the timer expires, then we increment the subscription's silent push count.
static constexpr Seconds silentPushTimeoutForProduction { 30_s };
static constexpr Seconds silentPushTimeoutForTesting { 1_s };

constexpr auto protocolVersionKey = "protocol version"_s;
constexpr uint64_t protocolVersionValue = 5;
constexpr auto protocolEncodedMessageKey = "encoded message"_s;

// FIXME: ConnectionToMachService traits requires we have a message type, so keep this placeholder here
// until we can remove that requirement.
enum class MessageType : uint8_t {
    EchoTwice
};

static constexpr unsigned long pushActionSetting = 0x8054000;

#ifdef __OBJC__
inline NSString *pushActionVersionKey()
{
    return @"WebPushActionVersion";
}

inline NSNumber *currentPushActionVersion()
{
    return @1;
}

inline NSString *pushActionPartitionKey()
{
    return @"WebPushActionPartition";
}

inline NSString *pushActionTypeKey()
{
    return @"WebPushActionType";
}
#endif // __OBJC__

} // namespace WebKit::WebPushD

#endif // __cplusplus
