/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 17, 2023.
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
#import "config.h"
#import "PrivateClickMeasurementConnection.h"

#import "DaemonEncoder.h"
#import "PrivateClickMeasurementXPCUtilities.h"
#import <wtf/NeverDestroyed.h>

namespace WebKit::PCM {

void Connection::newConnectionWasInitialized() const
{
    ASSERT(m_connection);
    if (!m_networkSession
        || m_networkSession->sessionID().isEphemeral()
        || !m_networkSession->privateClickMeasurementDebugModeEnabled())
        return;

    Daemon::Encoder encoder;
    encoder.encode(true);
    send(MessageType::SetDebugModeIsEnabled, encoder.takeBuffer());
}

void Connection::connectionReceivedEvent(xpc_object_t request)
{
    if (xpc_get_type(request) != XPC_TYPE_DICTIONARY)
        return;
    String debugMessage = xpc_dictionary_get_wtfstring(request, protocolDebugMessageKey);
    if (!debugMessage)
        return;
    auto messageLevel = static_cast<JSC::MessageLevel>(xpc_dictionary_get_uint64(request, protocolDebugMessageLevelKey));
    auto* networkSession = m_networkSession.get();
    if (!networkSession)
        return;
    m_networkSession->networkProcess().broadcastConsoleMessage(m_networkSession->sessionID(), MessageSource::PrivateClickMeasurement, messageLevel, debugMessage);
}

OSObjectPtr<xpc_object_t> Connection::dictionaryFromMessage(MessageType messageType, EncodedMessage&& message) const
{
    auto dictionary = adoptOSObject(xpc_dictionary_create(nullptr, nullptr, 0));
    addVersionAndEncodedMessageToDictionary(WTFMove(message), dictionary.get());
    xpc_dictionary_set_uint64(dictionary.get(), protocolMessageTypeKey, static_cast<uint64_t>(messageType));
    return dictionary;
}

} // namespace WebKit::PCM
