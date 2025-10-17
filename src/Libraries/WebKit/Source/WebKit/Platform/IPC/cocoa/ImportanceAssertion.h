/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 20, 2023.
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

#if PLATFORM(MAC)

#include <mach/message.h>

namespace IPC {

class ImportanceAssertion {
public:
    ImportanceAssertion() = default;

    explicit ImportanceAssertion(mach_msg_header_t* header)
    {
        if (MACH_MSGH_BITS_HAS_VOUCHER(header->msgh_bits)) {
            m_voucher = std::exchange(header->msgh_voucher_port, MACH_VOUCHER_NULL);
            header->msgh_bits &= ~(MACH_MSGH_BITS_VOUCHER_MASK | MACH_MSGH_BITS_RAISEIMP);
        }
    }

    ImportanceAssertion(ImportanceAssertion&& other)
        : m_voucher(std::exchange(other.m_voucher, MACH_VOUCHER_NULL))
    {
    }

    ImportanceAssertion& operator=(ImportanceAssertion&& other)
    {
        if (&other != this)
            std::swap(m_voucher, other.m_voucher);
        return *this;
    }

    ImportanceAssertion(const ImportanceAssertion&) = delete;
    ImportanceAssertion& operator=(const ImportanceAssertion&) = delete;

    ~ImportanceAssertion()
    {
        if (!m_voucher)
            return;

        kern_return_t kr = mach_voucher_deallocate(m_voucher);
        ASSERT_UNUSED(kr, !kr);
        m_voucher = MACH_VOUCHER_NULL;
    }

private:
    mach_voucher_t m_voucher { MACH_VOUCHER_NULL };
};

}

#endif // PLATFORM(MAC)
