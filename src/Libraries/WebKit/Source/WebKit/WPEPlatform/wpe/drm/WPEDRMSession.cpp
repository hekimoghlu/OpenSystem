/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 10, 2024.
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
#include "WPEDRMSession.h"

#include "WPEDRMSessionLogind.h"
#include <fcntl.h>
#include <unistd.h>
#include <wtf/TZoneMallocInlines.h>

namespace WPE {

namespace DRM {

WTF_MAKE_TZONE_ALLOCATED_IMPL(Session);

std::unique_ptr<Session> Session::create()
{
#if ENABLE(JOURNALD_LOG)
    if (auto session = SessionLogind::create())
        return session;
#endif

    return makeUnique<Session>();
}

int Session::openDevice(const char* path, int flags)
{
    return open(path, flags | O_CLOEXEC);
}

int Session::closeDevice(int fd)
{
    return close(fd);
}

} // namespace DRM

} // namespace WPE

