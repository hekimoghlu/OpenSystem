/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 18, 2023.
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
#ifndef __DISKARBITRATIOND_DAMOUNT__
#define __DISKARBITRATIOND_DAMOUNT__

#include <CoreFoundation/CoreFoundation.h>

#include "DADisk.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

enum
{
    kDAMountPointActionLink,
    kDAMountPointActionMake,
    kDAMountPointActionMove,
    kDAMountPointActionNone
};

typedef UInt32 DAMountPointAction;

enum
{
    kDAMountPreferenceDefer,
    kDAMountPreferenceTrust,
    kDAMountPreferenceDisableAutoMount,
    kDAMountPreferenceEnableUserFSMount
};

typedef UInt32 DAMountPreference;

typedef void ( *DAMountCallback )( int status, CFURLRef mountpoint, void * context );

extern void DAMount( DADiskRef       disk,
                     CFURLRef        mountpoint,
                     DAMountCallback callback,
                     void *          callbackContext );

extern void DAMountWithArguments( DADiskRef       disk,
                                  CFURLRef        mountpoint,
                                  DAMountCallback callback,
                                  void *          callbackContext,
                                  ... );

extern Boolean DAMountContainsArgument( CFStringRef arguments, CFStringRef argument );

extern CFURLRef DAMountCreateMountPoint( DADiskRef disk );

extern CFURLRef DAMountCreateMountPointWithAction( DADiskRef disk, DAMountPointAction action );

extern Boolean DAMountGetPreference( DADiskRef disk, DAMountPreference preference );

extern void DAMountRemoveMountPoint( CFURLRef mountpoint );

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* !__DISKARBITRATIOND_DAMOUNT__ */
