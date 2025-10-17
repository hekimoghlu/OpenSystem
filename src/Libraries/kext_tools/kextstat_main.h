/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 17, 2022.
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
#ifndef _KEXTSTAT_MAIN_H
#define _KEXTSTAT_MAIN_H

#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/kext/OSKext.h>

#include <getopt.h>
#include <sysexits.h>

#include <mach-o/arch.h>

#include "kext_tools_util.h"

#pragma mark Basic Types & Constants
/*******************************************************************************
* Constants
*******************************************************************************/
enum {
    kKextstatExitOK          = EX_OK,
    kKextstatExitError  = 1,

    // don't think we use it
    kKextstatExitUnspecified = 11,

    // don't actually exit with this, it's just a sentinel value
    kKextstatExitHelp        = 33
};

#pragma mark Command-line Option Definitions
/*******************************************************************************
* Command-line options. This data is used by getopt_long_only().
*
* Options common to all kext tools are in kext_tools_util.h.
*******************************************************************************/

#define kOptNameNoKernelComponents  "no-kernel"
#define kOptNameListOnly            "list-only"
#define kOptNameArchitecture        "arch"
#define kOptNameSort        "sort"

#define kOptNoKernelComponents      'k'
#define kOptListOnly                'l'
#define kOptArchitecture            'a'
#define kOptSort                    's'

#if 0
// bundle-id,version,compatible-version,is-kernel,is-interface,retaincount,path,uuid,started,prelinked,index,address,size,wired,dependencies,classes,cputype,cpusubtype
CFBundleIdentifier
CFBundleVersion
kOSBundleCompatibleVersionKey
kOSKernelResourceKey
kOSBundleIsInterfaceKey
kOSBundlePathKey
kOSBundleUUIDKey
kOSBundleStartedKey
kOSBundlePrelinkedKey
kOSBundleLoadTagKey
kOSBundleLoadAddressKey
kOSBundleLoadSizeKey
kOSBundleWiredSizeKey
kOSBundleDependenciesKey
kOSBundleMetaClassesKey
#endif

#define kOptChars                "b:hkla"

/* Options with no single-letter variant.  */
// Do not use -1, that's getopt() end-of-args return value
// and can cause confusion
#define kLongOptLongindexHack (-2)

static int longopt = 0;

static struct option sOptInfo[] = {
    { kOptNameHelp,               no_argument,        NULL,     kOptHelp },
    { kOptNameBundleIdentifier,   required_argument,  NULL,     kOptBundleIdentifier },
    { kOptNameNoKernelComponents, no_argument,        NULL,     kOptNoKernelComponents },
    { kOptNameListOnly,           no_argument,        NULL,     kOptListOnly },
    { kOptNameSort,               no_argument,        NULL,     kOptSort },
    { kOptNameArch,               no_argument,        NULL,     kOptArchitecture },

    { NULL, 0, NULL, 0 }  // sentinel to terminate list
};

#pragma mark Tool Args Structure
/*******************************************************************************
* Tool Args Structure
*******************************************************************************/
typedef struct {
    Boolean            flagNoKernelComponents;
    Boolean            flagListOnly;
    Boolean            flagShowArchitecture;
    Boolean            flagSortByLoadAddress;
    CFMutableArrayRef  bundleIDs;          // must release

    CFDictionaryRef    loadedKextInfo;     // must release
    const NXArchInfo * runningKernelArch;  // do not free
} KextstatArgs;

#pragma mark Function Prototypes
/*******************************************************************************
* Function Prototypes
*******************************************************************************/
ExitStatus readArgs(int argc, char * const * argv, KextstatArgs * toolArgs);
void printKextInfo(CFDictionaryRef kextInfo, KextstatArgs * toolArgs);

Boolean getNumValue(CFNumberRef aNumber, CFNumberType type, void * valueOut);
int compareKextInfo(const void * vKextInfo1, const void * vKextInfo2);
int compareKextInfoLoadAddress(const void * vKextInfo1, const void * vKextInfo2);

CFComparisonResult compareNumbers(
    const void * val1,
    const void * val2,
          void * context);

void usage(UsageLevel usageLevel);

#endif /* _KEXTSTAT_MAIN_H */
