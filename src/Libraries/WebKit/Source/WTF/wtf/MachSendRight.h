/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 16, 2024.
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

#if PLATFORM(COCOA)

#include <mach/mach_port.h>

namespace WTF {

class MachSendRight {
    WTF_MAKE_FAST_ALLOCATED;
public:
    WTF_EXPORT_PRIVATE static MachSendRight adopt(mach_port_t);
    WTF_EXPORT_PRIVATE static MachSendRight create(mach_port_t);
    WTF_EXPORT_PRIVATE static MachSendRight createFromReceiveRight(mach_port_t);

    MachSendRight() = default;
    WTF_EXPORT_PRIVATE explicit MachSendRight(const MachSendRight&);
    WTF_EXPORT_PRIVATE MachSendRight(MachSendRight&&);
    WTF_EXPORT_PRIVATE ~MachSendRight();

    WTF_EXPORT_PRIVATE MachSendRight& operator=(MachSendRight&&);

    explicit operator bool() const { return m_port != MACH_PORT_NULL; }

    mach_port_t sendRight() const { return m_port; }

    WTF_EXPORT_PRIVATE mach_port_t leakSendRight() WARN_UNUSED_RETURN;

private:
    explicit MachSendRight(mach_port_t);

    mach_port_t m_port { MACH_PORT_NULL };
};

WTF_EXPORT_PRIVATE void deallocateSendRightSafely(mach_port_t);

}

using WTF::deallocateSendRightSafely;

#endif
