/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 8, 2023.
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

#if ENABLE(MODEL_PROCESS)

#include "SharedPreferencesForWebProcess.h"
#include <WebCore/ProcessIdentity.h>
#include <wtf/MachSendRight.h>

#if HAVE(AUDIT_TOKEN)
#include "CoreIPCAuditToken.h"
#endif

namespace WebKit {

struct ModelProcessConnectionParameters {
    WebCore::ProcessIdentity webProcessIdentity;
    SharedPreferencesForWebProcess sharedPreferencesForWebProcess;

#if ENABLE(IPC_TESTING_API)
    bool ignoreInvalidMessageForTesting { false };
#endif
#if HAVE(AUDIT_TOKEN)
    std::optional<CoreIPCAuditToken> presentingApplicationAuditToken;
#endif
};

} // namespace WebKit

#endif // ENABLE(MODEL_PROCESS)
