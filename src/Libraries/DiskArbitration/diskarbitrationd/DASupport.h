/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 20, 2021.
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
#ifndef __DISKARBITRATIOND_DASUPPORT__
#define __DISKARBITRATIOND_DASUPPORT__

#include <CoreFoundation/CoreFoundation.h>

#include "DADisk.h"
#include "DAInternal.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

typedef void ( *DAAuthorizeCallback )( DAReturn status, void * context );

extern DAReturn DAAuthorize( DASessionRef        session,
                             _DAAuthorizeOptions options,
                             DADiskRef           disk,
                             uid_t               userUID,
                             gid_t               userGID,
                             const char *        right );

extern void DAAuthorizeWithCallback( DASessionRef        session,
                                     _DAAuthorizeOptions options,
                                     DADiskRef           disk,
                                     uid_t               userUID,
                                     gid_t               userGID,
                                     DAAuthorizeCallback callback,
                                     void *              callbackContext,
                                     const char *        right );

extern const CFStringRef kDAFileSystemKey; /* ( DAFileSystem ) */

extern void DAFileSystemListRefresh( void );
#ifdef DA_FSKIT
extern void DAProbeWithFSKit( CFStringRef deviceName ,
                              CFStringRef bundleID ,
                              bool doFsck ,
                              DAFileSystemProbeCallback callback ,
                              void *callbackContext );

extern void DARepairWithFSKit( CFStringRef deviceName ,
                               CFStringRef bundleID ,
                               DAFileSystemCallback callback ,
                               void *callbackContext );

typedef void ( *DAProbeCandidateCallback )( CFDictionaryRef candidate ,
                                            void *probeCallbackContext );

extern void DACheckForFSKit( void );

extern void DAGetFSModulesForUser( uid_t user ,
                                   void *probeCallbackContext );

extern CFStringRef DAGetFSKitBundleID( CFStringRef filesystemName );
#endif

extern const CFStringRef kDAMountMapMountAutomaticKey; /* ( CFBoolean ) */
extern const CFStringRef kDAMountMapMountOptionsKey;   /* ( CFString  ) */
extern const CFStringRef kDAMountMapMountPathKey;      /* ( CFURL     ) */
extern const CFStringRef kDAMountMapProbeIDKey;        /* ( CFUUID    ) */
extern const CFStringRef kDAMountMapProbeKindKey;      /* ( CFString  ) */

extern void DAMountMapListRefresh1( void );
extern void DAMountMapListRefresh2( void );

extern const CFStringRef kDAPreferenceMountDeferExternalKey;         /* ( CFBoolean ) */
extern const CFStringRef kDAPreferenceMountDeferInternalKey;         /* ( CFBoolean ) */
extern const CFStringRef kDAPreferenceMountDeferRemovableKey;        /* ( CFBoolean ) */
extern const CFStringRef kDAPreferenceMountTrustExternalKey;         /* ( CFBoolean ) */
extern const CFStringRef kDAPreferenceMountTrustInternalKey;         /* ( CFBoolean ) */
extern const CFStringRef kDAPreferenceMountTrustRemovableKey;        /* ( CFBoolean ) */
extern const CFStringRef kDAPreferenceAutoMountDisableKey;           /* ( CFBoolean ) */
extern const CFStringRef kDAPreferenceEnableUserFSMountExternalKey;  /* ( CFBoolean ) */
extern const CFStringRef kDAPreferenceEnableUserFSMountInternalKey;  /* ( CFBoolean ) */
extern const CFStringRef kDAPreferenceEnableUserFSMountRemovableKey; /* ( CFBoolean ) */
extern const CFStringRef kDAPreferenceMountMethodkey;                 /* ( CFString ) */
extern const CFStringRef kDAPreferenceDisableEjectNotificationKey;   /* ( CFBoolean ) */
extern const CFStringRef kDAPreferenceDisableUnreadableNotificationKey; /* ( CFBoolean ) */
extern const CFStringRef kDAPreferenceDisableUnrepairableNotificationKey; /* ( CFBoolean ) */

extern void DAPreferenceListRefresh( void );

enum
{
///w:23678897:start
    _kDAUnitStateHasAPFS = 0x00000010,
///w:23678897:stop
    kDAUnitStateCommandActive        = 0x00000001,
    kDAUnitStateHasQuiesced          = 0x00000002,
    kDAUnitStateHasQuiescedNoTimeout = 0x00000004,
    kDAUnitStateStagedUnreadable     = 0x00010000
};

typedef UInt32 DAUnitState;

extern Boolean DAUnitGetState( DADiskRef disk, DAUnitState state );
extern Boolean DAUnitGetStateRecursively( DADiskRef disk, DAUnitState state );
extern void    DAUnitSetState( DADiskRef disk, DAUnitState state, Boolean value );

#if TARGET_OS_IOS
extern Boolean DADeviceIsUnlocked( void );
#endif

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* !__DISKARBITRATIOND_DASUPPORT__ */
