/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 21, 2022.
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
#include "ProcessIdentity.h"

#include "Logging.h"

#if HAVE(TASK_IDENTITY_TOKEN)
#include <mach/mach.h>
#endif

namespace WebCore {

ProcessIdentity::ProcessIdentity(CurrentProcessTag)
{
#if HAVE(TASK_IDENTITY_TOKEN)
    task_id_token_t identityToken;
    kern_return_t kr = task_create_identity_token(mach_task_self(), &identityToken);
    if (kr == KERN_SUCCESS)
        m_taskIdToken = MachSendRight::adopt(identityToken);
    else
        RELEASE_LOG_ERROR(Process, "task_create_identity_token() failed: %{private}s (%x)", mach_error_string(kr), kr);
#endif
}

ProcessIdentity::operator bool() const
{
#if HAVE(TASK_IDENTITY_TOKEN)
    return static_cast<bool>(m_taskIdToken);
#else
    return false;
#endif
}

#if HAVE(TASK_IDENTITY_TOKEN)
ProcessIdentity::ProcessIdentity(MachSendRight&& taskIdToken)
    : m_taskIdToken(WTFMove(taskIdToken))
{
}
#endif

ProcessIdentity& ProcessIdentity::operator=(const ProcessIdentity& other)
{
#if HAVE(TASK_IDENTITY_TOKEN)
    m_taskIdToken = MachSendRight { other.m_taskIdToken };
#endif
    UNUSED_PARAM(other);
    return *this;
}

}
