/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 28, 2024.
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

#include <WebCore/FileHandle.h>
#include <fcntl.h>
#include <wtf/FileSystem.h>

namespace IPC {

class Decoder;
class Encoder;

class SharedFileHandle {
public:
    static std::optional<SharedFileHandle> create(FileSystem::PlatformFileHandle);

#if PLATFORM(COCOA)
    explicit SharedFileHandle(MachSendRight&&);
    MachSendRight toMachSendRight() const;
#endif

    SharedFileHandle() = default;
    WebCore::FileHandle release() { return std::exchange(m_handle, { }); }

private:
    explicit SharedFileHandle(FileSystem::PlatformFileHandle handle)
        : m_handle(handle)
    {
    }

    WebCore::FileHandle m_handle;
};

} // namespace IPC


