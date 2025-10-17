/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 15, 2024.
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

#include "RTCDataChannelIdentifier.h"
#include "ScriptExecutionContextIdentifier.h"
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class RTCDataChannelRemoteHandler;

class RTCDataChannelRemoteHandlerConnection : public ThreadSafeRefCounted<RTCDataChannelRemoteHandlerConnection, WTF::DestructionThread::Main> {
public:
    virtual ~RTCDataChannelRemoteHandlerConnection() = default;

    virtual void connectToSource(RTCDataChannelRemoteHandler&, std::optional<ScriptExecutionContextIdentifier>, RTCDataChannelIdentifier, RTCDataChannelIdentifier) = 0;
    virtual void sendData(RTCDataChannelIdentifier, bool isRaw, std::span<const uint8_t>) = 0;
    virtual void close(RTCDataChannelIdentifier) = 0;
};

} // namespace WebCore
