/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 19, 2022.
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
#ifndef __DISKARBITRATIOND_DASESSION__
#define __DISKARBITRATIOND_DASESSION__

#include <CoreFoundation/CoreFoundation.h>
#include <DiskArbitration/DiskArbitration.h>
#include <DiskArbitration/DiskArbitrationPrivate.h>
#if TARGET_OS_OSX
#include <Security/Authorization.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

enum
{
    kDASessionOptionNoTimeout = 0x01000000
};

typedef UInt32 DASessionOption;

typedef UInt32 DASessionOptions;

enum
{
    kDASessionStateIdle    = 0x00000001,
    kDASessionStateTimeout = 0x01000000,
    kDASessionStateZombie  = 0x10000000
};

typedef UInt32 DASessionState;

typedef struct __DACallback * DACallbackRef;

typedef struct __DASession * DASessionRef;

///w:start
extern const char * _DASessionGetName( DASessionRef session );
///w:stop
extern DASessionRef      DASessionCreate( CFAllocatorRef allocator, const char * _name, pid_t _pid );
#if TARGET_OS_OSX
extern AuthorizationRef  DASessionGetAuthorization( DASessionRef session );
#endif
extern CFMutableArrayRef DASessionGetCallbackQueue( DASessionRef session );
extern CFMutableArrayRef DASessionGetCallbackRegister( DASessionRef session );
extern mach_port_t       DASessionGetID( DASessionRef session );
extern Boolean           DASessionGetIsFSKitd( DASessionRef session );
extern Boolean           DASessionGetOption( DASessionRef session, DASessionOption option );
extern DASessionOptions  DASessionGetOptions( DASessionRef session );
extern mach_port_t       DASessionGetServerPort( DASessionRef session );
extern Boolean           DASessionGetState( DASessionRef session, DASessionState state );
extern CFTypeID          DASessionGetTypeID( void );
extern Boolean           DASessionGetKeepAlive( DASessionRef session );
extern void              DASessionInitialize( void );
extern void              DASessionQueueCallback( DASessionRef session, DACallbackRef callback );
extern void              DASessionRegisterCallback( DASessionRef session, DACallbackRef callback );
#if TARGET_OS_OSX
extern void              DASessionSetAuthorization( DASessionRef session, AuthorizationRef authorization );
#endif
extern void              DASessionSetClientPort( DASessionRef session, mach_port_t client );
#ifdef DA_FSKIT
extern void              DASessionSetIsFSKitd( DASessionRef session, Boolean value );
#endif
extern void              DASessionSetOption( DASessionRef session, DASessionOption option, Boolean value );
extern void              DASessionSetOptions( DASessionRef session, DASessionOptions options, Boolean value );
extern void              DASessionSetState( DASessionRef session, DASessionState state, Boolean value );
extern void              DASessionSetKeepAlive( DASessionRef session , bool value);
extern void              DASessionUnregisterCallback( DASessionRef session, DACallbackRef callback );
extern void              DASessionCancelChannel( DASessionRef session );
extern void              DASessionScheduleWithDispatch( DASessionRef session );

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* !__DISKARBITRATIOND_DASESSION__ */
