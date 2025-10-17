/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 26, 2023.
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
#include <wtf/MachSendRight.h>

#include <mach/mach_error.h>
#include <mach/mach_init.h>
#include <utility>

#include <wtf/Logging.h>

namespace WTF {

static void retainSendRight(mach_port_t port)
{
    if (port == MACH_PORT_NULL)
        return;

    auto kr = KERN_SUCCESS;
    if (port != MACH_PORT_DEAD)
        kr = mach_port_mod_refs(mach_task_self(), port, MACH_PORT_RIGHT_SEND, 1);

    if (kr == KERN_INVALID_RIGHT || port == MACH_PORT_DEAD)
        kr = mach_port_mod_refs(mach_task_self(), port, MACH_PORT_RIGHT_DEAD_NAME, 1);

    if (kr != KERN_SUCCESS) {
        LOG_ERROR("mach_port_mod_refs error for port %d: %s (%x)", port, mach_error_string(kr), kr);
        if (kr == KERN_INVALID_RIGHT)
            CRASH();
    }
}

void deallocateSendRightSafely(mach_port_t port)
{
    if (port == MACH_PORT_NULL)
        return;

    auto kr = mach_port_deallocate(mach_task_self(), port);
    if (kr == KERN_SUCCESS)
        return;

    RELEASE_LOG_ERROR(Process, "mach_port_deallocate error for port %d: %{private}s (%#x)", port, mach_error_string(kr), kr);
    if (kr == KERN_INVALID_RIGHT || kr == KERN_INVALID_NAME)
        CRASH();
}

static void assertSendRight(mach_port_t port)
{
    if (port == MACH_PORT_NULL)
        return;

    unsigned count = 0;
    auto kr = mach_port_get_refs(mach_task_self(), port, MACH_PORT_RIGHT_SEND, &count);
    if (kr == KERN_SUCCESS && !count)
        kr = mach_port_get_refs(mach_task_self(), port, MACH_PORT_RIGHT_DEAD_NAME, &count);

    if (kr == KERN_SUCCESS && count > 0)
        return;

    RELEASE_LOG_ERROR(Process, "mach_port_get_refs error for port %d: %{private}s (%#x)", port, mach_error_string(kr), kr);
    CRASH();
}

MachSendRight MachSendRight::adopt(mach_port_t port)
{
    assertSendRight(port);
    return MachSendRight(port);
}

MachSendRight MachSendRight::create(mach_port_t port)
{
    retainSendRight(port);
    return adopt(port);
}

MachSendRight MachSendRight::createFromReceiveRight(mach_port_t receiveRight)
{
    ASSERT(MACH_PORT_VALID(receiveRight));
    if (mach_port_insert_right(mach_task_self(), receiveRight, receiveRight, MACH_MSG_TYPE_MAKE_SEND) == KERN_SUCCESS)
        return MachSendRight { receiveRight };
    return { };
}

MachSendRight::MachSendRight(mach_port_t port)
    : m_port(port)
{
}

MachSendRight::MachSendRight(MachSendRight&& other)
    : m_port(other.leakSendRight())
{
}

MachSendRight::MachSendRight(const MachSendRight& other)
    : m_port(other.m_port)
{
    retainSendRight(m_port);
}

MachSendRight::~MachSendRight()
{
    deallocateSendRightSafely(m_port);
}

MachSendRight& MachSendRight::operator=(MachSendRight&& other)
{
    if (this != &other) {
        deallocateSendRightSafely(m_port);
        m_port = other.leakSendRight();
    }

    return *this;
}

mach_port_t MachSendRight::leakSendRight()
{
    return std::exchange(m_port, MACH_PORT_NULL);
}

}
