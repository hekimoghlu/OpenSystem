/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 11, 2025.
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

#include <optional>

#if HAVE(TASK_IDENTITY_TOKEN)
#include <wtf/ArgumentCoder.h>
#include <wtf/MachSendRight.h>
#else
#include <variant>
#endif

namespace WebCore {

// Object to access proof of process identity.
// ProcessIdentifier identifies a process.
// ProcessIdentity grants access to the identity.
// Empty ProcessIdentity does not do anything.
class ProcessIdentity {
public:
    // Creates an process identity for current process or empty on error.
    enum CurrentProcessTag { CurrentProcess };
    WEBCORE_EXPORT explicit ProcessIdentity(CurrentProcessTag);

    // Creates an empty process identity that does not grant any access.
    ProcessIdentity() = default;
    WEBCORE_EXPORT ProcessIdentity(const ProcessIdentity&) = default;

    // Returns true for a process identity or false on empty identity.
    WEBCORE_EXPORT operator bool() const;

    WEBCORE_EXPORT ProcessIdentity& operator=(const ProcessIdentity&);

#if HAVE(TASK_IDENTITY_TOKEN)
    task_id_token_t taskIdToken() const { return m_taskIdToken.sendRight(); }
    const MachSendRight& taskId() const { return m_taskIdToken; }
#endif

private:
#if HAVE(TASK_IDENTITY_TOKEN)
    friend struct IPC::ArgumentCoder<ProcessIdentity, void>;
    WEBCORE_EXPORT ProcessIdentity(MachSendRight&& taskIdToken);
    MachSendRight m_taskIdToken;
#endif
};

}
