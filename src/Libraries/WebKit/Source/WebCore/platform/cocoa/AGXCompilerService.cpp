/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 3, 2025.
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
#include "AGXCompilerService.h"

#if PLATFORM(IOS_FAMILY)

#include <sys/utsname.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/text/ASCIILiteral.h>

namespace WebCore {

static std::optional<bool> hasAGXCompilerService;

void setDeviceHasAGXCompilerServiceForTesting()
{
    hasAGXCompilerService = true;
}

bool deviceHasAGXCompilerService()
{
    if (!hasAGXCompilerService) {
        struct utsname systemInfo;
        if (uname(&systemInfo)) {
            hasAGXCompilerService = false;
            return *hasAGXCompilerService;
        }
        auto machine = unsafeSpan(systemInfo.machine);
        if (equalSpans(machine, "iPad5,1"_span) || equalSpans(machine, "iPad5,2"_span) || equalSpans(machine, "iPad5,3"_span) || equalSpans(machine, "iPad5,4"_span))
            hasAGXCompilerService = true;
        else
            hasAGXCompilerService = false;
    }
    return *hasAGXCompilerService;
}

std::span<const ASCIILiteral> agxCompilerServices()
{
    static constexpr std::array services {
        "com.apple.AGXCompilerService"_s,
        "com.apple.AGXCompilerService-S2A8"_s
    };
    return services;
}

std::span<const ASCIILiteral> agxCompilerClasses()
{
    static constexpr std::array classes {
        "AGXCommandQueue"_s,
        "AGXDevice"_s,
        "AGXSharedUserClient"_s,
        "IOAccelContext"_s,
        "IOAccelContext2"_s,
        "IOAccelDevice"_s,
        "IOAccelDevice2"_s,
        "IOAccelSharedUserClient"_s,
        "IOAccelSharedUserClient2"_s,
        "IOAccelSubmitter2"_s,
    };
    return classes;
}

}

#endif
