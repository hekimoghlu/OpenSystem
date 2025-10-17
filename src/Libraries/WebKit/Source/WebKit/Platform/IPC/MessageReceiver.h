/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 6, 2022.
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

#include <wtf/AbstractRefCounted.h>
#include <wtf/Assertions.h>
#include <wtf/WeakPtr.h>

namespace IPC {

class Connection;
class Decoder;
class Encoder;

class MessageReceiver : public CanMakeWeakPtr<MessageReceiver>, public AbstractRefCounted {
public:
    virtual ~MessageReceiver()
    {
        ASSERT(!m_messageReceiverMapCount);
    }

    virtual void didReceiveMessage(Connection&, Decoder&)
    {
        ASSERT_NOT_REACHED();
    }

    virtual void didReceiveMessageWithReplyHandler(Decoder&, Function<void(UniqueRef<IPC::Encoder>&&)>&&)
    {
        ASSERT_NOT_REACHED();
    }

    virtual bool didReceiveSyncMessage(Connection&, Decoder&, UniqueRef<Encoder>&)
    {
        ASSERT_NOT_REACHED();
        return false;
    }

private:
    friend class MessageReceiverMap;

    void willBeAddedToMessageReceiverMap()
    {
#if ASSERT_ENABLED
        m_messageReceiverMapCount++;
#endif
    }

    void willBeRemovedFromMessageReceiverMap()
    {
        ASSERT(m_messageReceiverMapCount);
#if ASSERT_ENABLED
        m_messageReceiverMapCount--;
#endif
    }

#if ASSERT_ENABLED
    unsigned m_messageReceiverMapCount { 0 };
#endif
};

} // namespace IPC
