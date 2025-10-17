/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 17, 2024.
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

#include <wtf/HashMap.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/CString.h>

namespace IPC {

class Connection;
class Encoder;
class Decoder;
class MessageReceiver;
enum class ReceiverName : uint8_t;

class MessageReceiverMap {
public:
    MessageReceiverMap();
    ~MessageReceiverMap();

    void addMessageReceiver(ReceiverName, MessageReceiver&);
    void addMessageReceiver(ReceiverName, uint64_t destinationID, MessageReceiver&);

    void removeMessageReceiver(ReceiverName);
    void removeMessageReceiver(ReceiverName, uint64_t destinationID);
    void removeMessageReceiver(MessageReceiver&);

    void invalidate();

    bool dispatchMessage(Connection&, Decoder&);
    bool dispatchSyncMessage(Connection&, Decoder&, UniqueRef<Encoder>&);

private:
    // Message receivers that don't require a destination ID.
    HashMap<ReceiverName, WeakPtr<MessageReceiver>, WTF::IntHash<ReceiverName>, WTF::StrongEnumHashTraits<ReceiverName>> m_globalMessageReceivers;

    HashMap<std::pair<ReceiverName, uint64_t>, WeakPtr<MessageReceiver>, DefaultHash<std::pair<ReceiverName, uint64_t>>, PairHashTraits<WTF::StrongEnumHashTraits<ReceiverName>, HashTraits<uint64_t>>> m_messageReceivers;
};

};

namespace WTF {

template<typename T> struct DefaultHash;
template<> struct DefaultHash<IPC::ReceiverName> : IntHash<IPC::ReceiverName> { };

}
