/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 20, 2025.
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
#include "ArgumentCoders.h"
#include "Attachment.h"
#include "Decoder.h"
#include "Encoder.h"
#include "SharedFileHandle.h"
#include <wtf/MachSendRight.h>

#include <pal/spi/cocoa/FilePortSPI.h>

namespace IPC {

std::optional<SharedFileHandle> SharedFileHandle::create(FileSystem::PlatformFileHandle handle)
{
    return SharedFileHandle { handle };
}

SharedFileHandle::SharedFileHandle(MachSendRight&& fileport)
{
    int fd = fileport_makefd(fileport.sendRight());
    if (fd == -1)
        return;
    m_handle = WebCore::FileHandle { fd };
}

MachSendRight SharedFileHandle::toMachSendRight() const
{
    mach_port_name_t fileport = MACH_PORT_NULL;
    if (fileport_makeport(m_handle.handle(), &fileport) == -1)
        return { };

    return MachSendRight::adopt(fileport);
}

} // namespace IPC
