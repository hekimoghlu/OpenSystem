/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 19, 2025.
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

//
//  IOGDiagnoseUtils.cpp
//  IOGDiagnoseUtils
//
//  Created by JÃ©rÃ©my Tran on 8/8/17.
//

#include "IOGDiagnoseUtils.hpp"

#include <cstdint>
#include <cstdio>
#include <memory>

#include <mach/mach_error.h>

// Pick up local headers
#include "IOGraphicsTypes.h"

#ifdef TARGET_CPU_X86_64

#define COUNT_OF(x) \
    ((sizeof(x)/sizeof(0[x])) / ((size_t)(!(sizeof(x) % sizeof(0[x])))))
#define kResourcePath     kIOServicePlane ":/" kIOResourcesClass "/"

namespace {
kern_return_t openWranglerUC(
        IOConnect* diagConnectP, uint32_t type, const char **errmsgP)
{
    static const char * const sWranglerPath = kResourcePath "IODisplayWrangler";
    char errbuf[128] = "";
    kern_return_t err = kIOReturnInternalError;

    do {
        err = kIOReturnNotFound;
        IOObject wrangler(
                IORegistryEntryFromPath(kIOMasterPortDefault, sWranglerPath));
        if (!static_cast<bool>(wrangler)) {
            snprintf(errbuf, sizeof(errbuf),
                     "IODisplayWrangler '%s' search failed", sWranglerPath);
            continue;
        }

        IOConnect wranglerConnect(wrangler, type);
        err = wranglerConnect.err();
        if (err) {
            const char * typeStr
                = (type == kIOGDiagnoseConnectType) ? "Diagnose"
                : (type == kIOGDiagnoseGTraceType)  ? "Gtrace"
                : "Unknown";
            snprintf(errbuf, sizeof(errbuf),
                     "IOServiceOpen(%s) on IODisplayWrangler failed", typeStr);
            continue;
        }
        *diagConnectP = std::move(wranglerConnect);
    } while(false);

    if (err && errmsgP) {
        char *tmpMsg;
        asprintf(&tmpMsg, "%s - %s(%x)", errbuf, mach_error_string(err), err);
        *errmsgP = tmpMsg;
    }
    return err;
}
} // namespace

kern_return_t openDiagnostics(IOConnect* diagConnectP, const char **errmsgP)
{
    return openWranglerUC(diagConnectP, kIOGDiagnoseConnectType, errmsgP);
}

kern_return_t openGTrace(IOConnect* gtraceConnectP, const char **errmsgP)
{
    return openWranglerUC(gtraceConnectP, kIOGDiagnoseGTraceType, errmsgP);
}

#endif // TARGET_CPU_X86_64
