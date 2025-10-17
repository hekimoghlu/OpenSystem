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
#include <dispatch/dispatch.h>
#include <AssertMacros.h>
#include <strings.h>
#include <os/variant_private.h>
#include <sys/sysctl.h>

#include "debugging.h"
#include "SecInternalReleasePriv.h"

bool SecAreQARootCertificatesEnabled(void) {
    static bool sQACertsEnabled = false;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        int value = 0;
        size_t size = sizeof(value);
        int ret = sysctlbyname("security.mac.amfi.qa_root_certs_allowed", &value, &size, NULL, 0);
        if (ret == 0) {
            sQACertsEnabled = (value == 1);
        } else {
            secerror("Unable to check QA certificate status: %d", ret);
        }
    });
    return sQACertsEnabled;
}

bool SecIsInternalRelease(void) {
    static bool isInternal = false;


    return isInternal;
}

