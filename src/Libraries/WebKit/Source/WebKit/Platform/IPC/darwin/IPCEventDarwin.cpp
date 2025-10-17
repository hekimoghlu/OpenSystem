/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 5, 2023.
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
#include "IPCEvent.h"

#include "Logging.h"
#include "MachUtilities.h"
#include <mach/mach_init.h>
#include <mach/mach_port.h>
#include <mach/message.h>
#include <wtf/StdLibExtras.h>

namespace IPC {

// Arbitrary message IDs that do not collide with Mach notification messages.
constexpr mach_msg_id_t inlineBodyMessageID = 0xdba0dba;

static void requestNoSenderNotifications(mach_port_t port, mach_port_t notify)
{
    mach_port_t previousNotificationPort = MACH_PORT_NULL;
    auto kr = mach_port_request_notification(mach_task_self(), port, MACH_NOTIFY_NO_SENDERS, 0, notify, MACH_MSG_TYPE_MAKE_SEND_ONCE, &previousNotificationPort);
    ASSERT(kr == KERN_SUCCESS);
    if (kr != KERN_SUCCESS) {
        // If mach_port_request_notification fails, 'previousNotificationPort' will be uninitialized.
        LOG_ERROR("mach_port_request_notification failed: (%x) %s", kr, mach_error_string(kr));
    } else
        deallocateSendRightSafely(previousNotificationPort);
}

static void requestNoSenderNotifications(mach_port_t port)
{
    requestNoSenderNotifications(port, port);
}

static void clearNoSenderNotifications(mach_port_t port)
{
    requestNoSenderNotifications(port, MACH_PORT_NULL);
}

void Signal::signal()
{
    mach_msg_header_t message;
    zeroBytes(message);
    message.msgh_remote_port = m_sendRight.sendRight();
    message.msgh_local_port = MACH_PORT_NULL;
    message.msgh_bits = MACH_MSGH_BITS(MACH_MSG_TYPE_COPY_SEND, 0);
    message.msgh_id = inlineBodyMessageID;

    auto ret = mach_msg(&message, MACH_SEND_MSG, sizeof(message), 0, MACH_PORT_NULL, MACH_MSG_TIMEOUT_NONE, MACH_PORT_NULL);
    if (ret != KERN_SUCCESS)
        RELEASE_LOG_ERROR(Process, "IPC::Signal::signal Could not send mach message, error %x", ret);
}

std::optional<EventSignalPair> createEventSignalPair()
{
    // Create the listening port.
    mach_port_t listeningPort = MACH_PORT_NULL;
    auto kr = mach_port_allocate(mach_task_self(), MACH_PORT_RIGHT_RECEIVE, &listeningPort);
    if (kr != KERN_SUCCESS) {
        RELEASE_LOG_ERROR(Process, "createEventSignalPair: Could not allocate mach port, error %x", kr);
        return std::nullopt;
    }
    if (!MACH_PORT_VALID(listeningPort)) {
        RELEASE_LOG_ERROR(Process, "createEventSignalPair: Could not allocate mach port, returned port was invalid");
        return std::nullopt;
    }

    setMachPortQueueLength(listeningPort, 1);
    auto sendRight = MachSendRight::createFromReceiveRight(listeningPort);
    requestNoSenderNotifications(listeningPort);

    return EventSignalPair { Event { listeningPort }, Signal { WTFMove(sendRight) } };
}

Event::~Event()
{
    if (m_receiveRight != MACH_PORT_NULL) {
        clearNoSenderNotifications(m_receiveRight);
        mach_port_mod_refs(mach_task_self(), m_receiveRight, MACH_PORT_RIGHT_RECEIVE, -1);
    }
}

typedef struct {
    mach_msg_header_t header;
    mach_msg_trailer_t trailer;
} ReceiveMessage;

bool Event::wait()
{
    ReceiveMessage receiveMessage;
    zeroBytes(receiveMessage);
    mach_msg_return_t ret = mach_msg(&receiveMessage.header, MACH_RCV_MSG, 0, sizeof(receiveMessage), m_receiveRight, MACH_MSG_TIMEOUT_NONE, MACH_PORT_NULL);
    if (ret != MACH_MSG_SUCCESS)
        return false;
    if (receiveMessage.header.msgh_id != inlineBodyMessageID)
        return false;
    return true;
}

bool Event::waitFor(Timeout timeout)
{
    ReceiveMessage receiveMessage;
    zeroBytes(receiveMessage);
    mach_msg_return_t ret = mach_msg(&receiveMessage.header, MACH_RCV_MSG | MACH_RCV_TIMEOUT, 0, sizeof(receiveMessage), m_receiveRight, timeout.secondsUntilDeadline().milliseconds(), MACH_PORT_NULL);
    if (ret != MACH_MSG_SUCCESS)
        return false;
    if (receiveMessage.header.msgh_id != inlineBodyMessageID)
        return false;
    return true;
}

} // namespace IPC

