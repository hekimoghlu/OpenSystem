/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 19, 2023.
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
#ifndef _KEXTD_MAIN_H
#define _KEXTD_MAIN_H

#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/kext/OSKext.h>

#include <getopt.h>
#include <sysexits.h>

#include "kext_tools_util.h"


#pragma mark Basic Types & Constants
/*******************************************************************************
* Basic Types & Constants
*******************************************************************************/
enum {
    kKextdExitOK          = EX_OK,
    kKextdExitError,
    kKextdExitSigterm,

    // don't actually exit with this, it's just a sentinel value
    kKextdExitHelp        = 33
};

typedef struct {
    mach_msg_header_t header;
    int signum;
} kextd_mach_msg_signal_t;


#define kAppleSetupDonePath       "/var/db/.AppleSetupDone"
#define kKextcacheDelayStandard   (60)
#define kKextcacheDelayFirstBoot  (60 * 5)

#define kReleaseKextsDelay  (30)

#pragma mark Command-line Option Definitions
/*******************************************************************************
* Command-line options. This data is used by getopt_long_only().
*
* Options common to all kext tools are in kext_tools_util.h.
*******************************************************************************/

#define kOptNameNoCaches      "no-caches"
#define kOptNameDebug         "debug"
#define kOptNameNoJettison    "no-jettison"

#define kOptNoCaches          'c'
#define kOptDebug             'd'
#define kOptSafeBoot          'x'

#define kOptChars             "cdhqvx"

/* Options with no single-letter variant.  */
// Do not use -1, that's getopt() end-of-args return value
// and can cause confusion
#define kLongOptLongindexHack (-2)

#pragma mark Tool Args Structure
/*******************************************************************************
* Tool Args Structure
*******************************************************************************/
typedef struct {
    Boolean            useRepositoryCaches;
    Boolean            debugMode;
    Boolean            safeBootMode;     // actual or simulated

    Boolean            firstBoot;
} KextdArgs;

extern CFArrayRef gRepositoryURLs;

#pragma mark Function Prototypes
/*******************************************************************************
* Function Prototypes
*******************************************************************************/
ExitStatus readArgs(int argc, char * const * argv, KextdArgs * toolArgs);

void       checkStartupMkext(KextdArgs * toolArgs);
Boolean    isNetboot(void);
void       sendActiveToKernel(void);
void       sendFinishedToKernel(void);
ExitStatus setUpServer(KextdArgs * toolArgs);

bool isBootRootActive(void);

void handleSignal(int signum);
void handleSignalInRunloop(
    CFMachPortRef  port,
        void     * msg,
        CFIndex    size,
        void     * info);

void readExtensions(void);
void scheduleReleaseExtensions(void);
void releaseExtensions(CFRunLoopTimerRef timer, void * context);
void rescanExtensions(void);

void enableNetworkAuthentication(void);

void usage(UsageLevel usageLevel);

#endif /* _KEXTD_MAIN_H */
