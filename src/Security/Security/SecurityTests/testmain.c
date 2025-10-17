/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 14, 2023.
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
#include <stdio.h>
#include <stdlib.h>

#include <unistd.h>

#include <regressions/test/testenv.h>

#include "testlist.h"
#include <regressions/test/testlist_begin.h>
#include "testlist.h"
#include <regressions/test/testlist_end.h>

#include <dispatch/dispatch.h>
#include <CoreFoundation/CFRunLoop.h>
#include "featureflags/affordance_featureflags.h"
#include "keychain/ckks/CKKS.h"

int main(int argc, char *argv[])
{
    //printf("Build date : %s %s\n", __DATE__, __TIME__);
    //printf("WARNING: If running those tests on a device with a passcode, DONT FORGET TO UNLOCK!!!\n");

    SecCKKSDisable();
    KCSharingSetChangeTrackingEnabled(false);

#if 0 && NO_SERVER
    SOSCloudKeychainServerInit();
#endif

#if TARGET_OS_IPHONE
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        int result = tests_begin(argc, argv);

        fflush(stderr);
        fflush(stdout);

        sleep(1);
        
        exit(result);
    });

    CFRunLoopRun();

    return 0;
#else
    int result = tests_begin(argc, argv);

    fflush(stdout);
    fflush(stderr);

    sleep(1);

    return result;
#endif
}
