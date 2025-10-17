/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 1, 2024.
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
#ifndef __DISKARBITRATIOND_DAFILESYSTEM__
#define __DISKARBITRATIOND_DAFILESYSTEM__

#include <sys/types.h>
#include <CoreFoundation/CoreFoundation.h>
#include <dispatch/dispatch.h>
#include <dispatch/private.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

typedef struct __DAFileSystem * DAFileSystemRef;
typedef struct __DAFileSystemContext __DAFileSystemContext;
// When adding filesystem argument strings, remember to check for them in DAMountContainsArgument()
extern const CFStringRef kDAFileSystemMountArgumentForce;
extern const CFStringRef kDAFileSystemMountArgumentNoDevice;
extern const CFStringRef kDAFileSystemMountArgumentDevice;
extern const CFStringRef kDAFileSystemMountArgumentNoExecute;
extern const CFStringRef kDAFileSystemMountArgumentNoOwnership;
extern const CFStringRef kDAFileSystemMountArgumentNoPermission;
extern const CFStringRef kDAFileSystemMountArgumentOwnership;
extern const CFStringRef kDAFileSystemMountArgumentNoSetUserID;
extern const CFStringRef kDAFileSystemMountArgumentSetUserID;
extern const CFStringRef kDAFileSystemMountArgumentNoWrite;
extern const CFStringRef kDAFileSystemMountArgumentUnion;
extern const CFStringRef kDAFileSystemMountArgumentUpdate;
extern const CFStringRef kDAFileSystemMountArgumentNoBrowse;
extern const CFStringRef kDAFileSystemMountArgumentSnapshot;
extern const CFStringRef kDAFileSystemMountArgumentNoFollow;

extern const CFStringRef kDAFileSystemUnmountArgumentForce;

typedef void ( *DAFileSystemCallback )( int status, void * context );

typedef void ( *DAFileSystemProbeCallback )( int status, int clean, CFStringRef name, CFStringRef type, CFUUIDRef uuid, void * context );

extern CFStringRef _DAFileSystemCopyNameAndUUID( DAFileSystemRef filesystem, CFURLRef mountpoint, uuid_t *volumeUUID);

extern CFUUIDRef _DAFileSystemCreateUUIDFromString( CFAllocatorRef allocator, CFStringRef string );

extern DAFileSystemRef DAFileSystemCreate( CFAllocatorRef allocator, CFURLRef path );

extern DAFileSystemRef DAFileSystemCreateFromProperties( CFAllocatorRef allocator, CFDictionaryRef properties );

extern dispatch_mach_t DAFileSystemCreateMachChannel( void );

extern CFStringRef DAFileSystemGetKind( DAFileSystemRef filesystem );

extern CFDictionaryRef DAFileSystemGetProbeList( DAFileSystemRef filesystem );

extern CFBooleanRef DAFileSystemIsFSModule( DAFileSystemRef filesystem );

extern Boolean DAFilesystemShouldMountWithUserFS( DAFileSystemRef filesystem ,
                                                  CFStringRef preferredMountMethod );

extern CFTypeID DAFileSystemGetTypeID( void );

extern void DAFileSystemInitialize( void );

extern void DAFileSystemMount( DAFileSystemRef      filesystem,
                               CFURLRef             device,
                               CFStringRef          volumeName,
                               CFURLRef             mountpoint,
                               uid_t                userUID,
                               gid_t                userGID,
                               DAFileSystemCallback callback,
                               void *               callbackContext,
                               CFStringRef          preferredMountMethod  );

extern void DAFileSystemMountWithArguments( DAFileSystemRef      filesystem,
                                            CFURLRef             device,
                                            CFStringRef          volumeName,
                                            CFURLRef             mountpoint,
                                            uid_t                userUID,
                                            gid_t                userGID,
                                            CFStringRef          preferredMountMethod,
                                            DAFileSystemCallback callback,
                                            void *               callbackContext,
                                            ... );

extern void DAFileSystemProbe( DAFileSystemRef           filesystem,
                               CFURLRef                  device,
                               const char *              deviceBSDPath,
                               const char *              containerBSDPath,
                               DAFileSystemProbeCallback callback,
                               void *                    callbackContext,
                               bool                      doFsck );

extern void DAFileSystemRename( DAFileSystemRef      filesystem,
                                CFURLRef             mountpoint,
                                CFStringRef          name,
                                DAFileSystemCallback callback,
                                void *               callbackContext );

extern void DAFileSystemRepair( DAFileSystemRef      filesystem,
                                CFURLRef             device,
                                int                  fd,
                                DAFileSystemCallback callback,
                                void *               callbackContext );

extern void DAFileSystemRepairQuotas( DAFileSystemRef      filesystem,
                                      CFURLRef             mountpoint,
                                      DAFileSystemCallback callback,
                                      void *               callbackContext );

extern void DAFileSystemUnmount( DAFileSystemRef      filesystem,
                                 CFURLRef             mountpoint,
                                 DAFileSystemCallback callback,
                                 void *               callbackContext );

extern void DAFileSystemUnmountWithArguments( DAFileSystemRef      filesystem,
                                              CFURLRef             mountpoint,
                                              DAFileSystemCallback callback,
                                              void *               callbackContext,
                                             ... );
                                             
#if TARGET_OS_OSX || TARGET_OS_IOS
extern int __DAMountUserFSVolume( void * parameter );
extern void __DAMountUserFSVolumeCallback( int status, void * parameter );
extern int DAUserFSOpen( char *path, int flags );
extern CFStringRef DSFSKitGetBundleNameWithoutSuffix( CFStringRef filesystemName );
#endif

struct __DAFileSystemContext
{
    DAFileSystemCallback callback;
    void *               callbackContext;
    CFStringRef          deviceName;
    CFUUIDRef            volumeUUID;
    CFStringRef          fileSystem;
    CFStringRef          mountPoint;
    CFStringRef          volumeName;
    CFStringRef          mountOptions;
};

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* !__DISKARBITRATIOND_DAFILESYSTEM__ */
