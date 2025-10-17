/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 30, 2021.
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

#include <wtf/Forward.h>
#include <wtf/ThreadSafeWeakPtr.h>

namespace WebCore {

class WebSocketChannelClient : public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<WebSocketChannelClient, WTF::DestructionThread::Main> {
public:
    virtual ~WebSocketChannelClient() = default;
    virtual void didConnect() = 0;
    virtual void didReceiveMessage(String&&) = 0;
    virtual void didReceiveBinaryData(Vector<uint8_t>&&) = 0;
    virtual void didReceiveMessageError(String&&) = 0;
    virtual void didUpdateBufferedAmount(unsigned bufferedAmount) = 0;
    virtual void didStartClosingHandshake() = 0;
    enum ClosingHandshakeCompletionStatus {
        ClosingHandshakeIncomplete,
        ClosingHandshakeComplete
    };
    virtual void didClose(unsigned unhandledBufferedAmount, ClosingHandshakeCompletionStatus, unsigned short code, const String& reason) = 0;
    virtual void didUpgradeURL() = 0;

protected:
    WebSocketChannelClient() = default;
};

} // namespace WebCore
