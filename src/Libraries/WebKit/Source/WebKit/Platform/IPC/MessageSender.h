/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 14, 2025.
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

#include <optional>
#include <wtf/Forward.h>

namespace IPC {

class Connection;
class Decoder;
class Encoder;
class Timeout;
enum class SendOption : uint8_t;
enum class SendSyncOption : uint8_t;
struct AsyncReplyIDType;
struct ConnectionAsyncReplyHandler;
template<typename> class ConnectionSendSyncResult;
using AsyncReplyID = AtomicObjectIdentifier<AsyncReplyIDType>;

class MessageSender {
public:
    virtual ~MessageSender();

    template<typename T> inline bool send(T&& message); // Defined in MessageSenderInlines.h.
    template<typename T> inline bool send(T&& message, OptionSet<SendOption>); // Defined in MessageSenderInlines.h.
    template<typename T> inline bool send(T&& message, uint64_t destinationID); // Defined in MessageSenderInlines.h.
    template<typename T> inline bool send(T&& message, uint64_t destinationID, OptionSet<SendOption>); // Defined in MessageSenderInlines.h.
    template<typename T, typename U, typename V, typename W> inline bool send(T&& message, ObjectIdentifierGeneric<U, V, W> destinationID);
    template<typename T, typename U, typename V, typename W> inline bool send(T&& message, ObjectIdentifierGeneric<U, V, W> destinationID, OptionSet<SendOption>);

    template<typename T> using SendSyncResult = ConnectionSendSyncResult<T>;
    template<typename T> inline SendSyncResult<T> sendSync(T&& message);
    template<typename T> inline SendSyncResult<T> sendSync(T&& message, Timeout);
    template<typename T> inline SendSyncResult<T> sendSync(T&& message, Timeout, OptionSet<SendSyncOption>);
    template<typename T> inline SendSyncResult<T> sendSync(T&& message, uint64_t destinationID);
    template<typename T> inline SendSyncResult<T> sendSync(T&& message, uint64_t destinationID, Timeout);
    template<typename T> inline SendSyncResult<T> sendSync(T&& message, uint64_t destinationID, Timeout, OptionSet<SendSyncOption>);
    template<typename T, typename U, typename V, typename W> inline SendSyncResult<T> sendSync(T&& message, ObjectIdentifierGeneric<U, V, W> destinationID);
    template<typename T, typename U, typename V, typename W> inline SendSyncResult<T> sendSync(T&& message, ObjectIdentifierGeneric<U, V, W> destinationID, Timeout);
    template<typename T, typename U, typename V, typename W> inline SendSyncResult<T> sendSync(T&& message, ObjectIdentifierGeneric<U, V, W> destinationID, Timeout, OptionSet<SendSyncOption>);

    using AsyncReplyID = IPC::AsyncReplyID;
    template<typename T, typename C> inline std::optional<AsyncReplyID> sendWithAsyncReply(T&& message, C&& completionHandler);
    template<typename T, typename C> inline std::optional<AsyncReplyID> sendWithAsyncReply(T&& message, C&& completionHandler, OptionSet<SendOption>);
    template<typename T, typename C> inline std::optional<AsyncReplyID> sendWithAsyncReply(T&& message, C&& completionHandler, uint64_t destinationID);
    template<typename T, typename C> inline std::optional<AsyncReplyID> sendWithAsyncReply(T&& message, C&& completionHandler, uint64_t destinationID, OptionSet<SendOption>);
    template<typename T, typename C, typename U, typename V, typename W> inline std::optional<AsyncReplyID> sendWithAsyncReply(T&& message, C&& completionHandler, ObjectIdentifierGeneric<U, V, W> destinationID);
    template<typename T, typename C, typename U, typename V, typename W> inline std::optional<AsyncReplyID> sendWithAsyncReply(T&& message, C&& completionHandler, ObjectIdentifierGeneric<U, V, W> destinationID, OptionSet<SendOption>);

    template<typename T> Ref<typename T::Promise> inline sendWithPromisedReply(T&& message);
    template<typename T> Ref<typename T::Promise> inline sendWithPromisedReply(T&& message, uint64_t destinationID);
    template<typename T> Ref<typename T::Promise> inline sendWithPromisedReply(T&& message, uint64_t destinationID, OptionSet<SendOption>);

    template<typename T> inline bool sendWithoutUsingIPCConnection(T&& message) const;
    virtual bool performSendWithoutUsingIPCConnection(UniqueRef<Encoder>&&) const;

    template<typename T, typename C> inline bool sendWithAsyncReplyWithoutUsingIPCConnection(T&& message, C&& completionHandler) const;
    virtual bool performSendWithAsyncReplyWithoutUsingIPCConnection(UniqueRef<Encoder>&&, CompletionHandler<void(Decoder*)>&&) const;

    virtual bool sendMessage(UniqueRef<Encoder>&&, OptionSet<SendOption>);

    using AsyncReplyHandler = ConnectionAsyncReplyHandler;
    virtual bool sendMessageWithAsyncReply(UniqueRef<Encoder>&&, AsyncReplyHandler, OptionSet<SendOption>);

private:
    virtual Connection* messageSenderConnection() const = 0;
    virtual uint64_t messageSenderDestinationID() const = 0;
};

} // namespace IPC
