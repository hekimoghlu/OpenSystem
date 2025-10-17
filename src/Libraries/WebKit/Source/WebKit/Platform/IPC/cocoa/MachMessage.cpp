/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 27, 2024.
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
#include "MachMessage.h"

#if PLATFORM(COCOA)

#include <mach/mach.h>

namespace IPC {

// Version of round_msg() using CheckedSize for extra safety.
static inline CheckedSize safeRoundMsg(CheckedSize value)
{
    constexpr size_t alignment = sizeof(natural_t);
    return ((value + (alignment - 1)) / alignment) * alignment;
}

std::unique_ptr<MachMessage> MachMessage::create(MessageName messageName, size_t size)
{
    auto bufferSize = CheckedSize(sizeof(MachMessage)) + size;
    if (bufferSize.hasOverflowed())
        return nullptr;
    void* memory = WTF::fastZeroedMalloc(bufferSize);
    return std::unique_ptr<MachMessage> { new (NotNull, memory) MachMessage { messageName, size } };
}

MachMessage::MachMessage(MessageName messageName, size_t size)
    : m_messageName { messageName }
    , m_size { size }
{
}

MachMessage::~MachMessage()
{
    if (m_shouldFreeDescriptors)
        ::mach_msg_destroy(header());
}

CheckedSize MachMessage::messageSize(size_t bodySize, size_t portDescriptorCount, size_t memoryDescriptorCount)
{
    CheckedSize messageSize = sizeof(mach_msg_header_t);
    messageSize += bodySize;

    if (portDescriptorCount || memoryDescriptorCount) {
        messageSize += sizeof(mach_msg_body_t);
        messageSize += (portDescriptorCount * sizeof(mach_msg_port_descriptor_t));
        messageSize += (memoryDescriptorCount * sizeof(mach_msg_ool_descriptor_t));
    }

    return safeRoundMsg(messageSize);
}

void MachMessage::leakDescriptors()
{
    m_shouldFreeDescriptors = false;
}

}

#endif // PLATFORM(COCOA)
