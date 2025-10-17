/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 11, 2024.
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
#import "config.h"
#import "SandboxUtilities.h"

#import <array>
#import <sys/param.h>
#import <wtf/OSObjectPtr.h>
#import <wtf/spi/darwin/SandboxSPI.h>
#import <wtf/spi/darwin/XPCSPI.h>
#import <wtf/text/WTFString.h>

namespace WebKit {

bool currentProcessIsSandboxed()
{
    return sandbox_check(getpid(), nullptr, SANDBOX_FILTER_NONE);
}

bool connectedProcessIsSandboxed(xpc_connection_t connectionToParent)
{
    audit_token_t token;
    xpc_connection_get_audit_token(connectionToParent, &token);
    return sandbox_check_by_audit_token(token, nullptr, SANDBOX_FILTER_NONE);
}

bool processHasContainer()
{
    static bool hasContainer = !pathForProcessContainer().isEmpty();
    return hasContainer;
}

String pathForProcessContainer()
{
    std::array<char, MAXPATHLEN> path;
    path[0] = 0;
    sandbox_container_path_for_pid(getpid(), path.data(), path.size());

    return String::fromUTF8(path.data());
}

}
