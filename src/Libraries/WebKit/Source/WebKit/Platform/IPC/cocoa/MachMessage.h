/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 8, 2024.
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

#if PLATFORM(COCOA)

#include "MessageNames.h"
#include <mach/message.h>
#include <memory>
#include <wtf/CheckedArithmetic.h>
#include <wtf/StdLibExtras.h>
#include <wtf/text/CString.h>

namespace IPC {

class MachMessage {
    WTF_MAKE_FAST_ALLOCATED;
public:
    static std::unique_ptr<MachMessage> create(MessageName, size_t);
    ~MachMessage();

    static CheckedSize messageSize(size_t bodySize, size_t portDescriptorCount, size_t memoryDescriptorCount);

    size_t size() const { return m_size; }
    mach_msg_header_t* header() { return m_messageHeader; }

    std::span<uint8_t> span() { return unsafeMakeSpan(reinterpret_cast<uint8_t*>(m_messageHeader), m_size); }

    void leakDescriptors();

    ReceiverName messageReceiverName() const { return receiverName(m_messageName); }
    MessageName messageName() const { return m_messageName; }

private:
    MachMessage(MessageName, size_t);

    MessageName m_messageName;
    size_t m_size;
    bool m_shouldFreeDescriptors { true };
    mach_msg_header_t m_messageHeader[];
};

}

#endif // PLATFORM(COCOA)
