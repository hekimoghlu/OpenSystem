/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 6, 2025.
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
#ifndef __DISKARBITRATIOND_DAQUEUE__
#define __DISKARBITRATIOND_DAQUEUE__

#include <CoreFoundation/CoreFoundation.h>

#include "DADisk.h"
#include "DADissenter.h"
#include "DARequest.h"
#include "DASession.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

typedef void ( *DAResponseCallback )( CFTypeRef response, void * context );

extern Boolean _DAResponseDispatch( CFTypeRef response, SInt32 responseID );

extern void DADiskAppearedCallback( DADiskRef disk );

extern void DADiskClaimReleaseCallback( DADiskRef disk, DACallbackRef callback, DAResponseCallback response, void * responseContext );

extern void DADiskDescriptionChangedCallback( DADiskRef disk, CFTypeRef key );

extern void DADiskDisappearedCallback( DADiskRef disk );

extern void DADiskEject( DADiskRef disk, DADiskEjectOptions options, DACallbackRef callback );

extern void DADiskEjectApprovalCallback( DADiskRef disk, DAResponseCallback response, void * responseContext );

extern void DADiskMount( DADiskRef disk, CFURLRef mountpoint, DADiskMountOptions options, DACallbackRef callback );

extern void DADiskMountApprovalCallback( DADiskRef disk, DAResponseCallback response, void * responseContext );

extern void DADiskMountWithArguments( DADiskRef disk, CFURLRef mountpoint, DADiskMountOptions options, DACallbackRef callback, CFStringRef arguments );

extern void DADiskPeekCallback( DADiskRef disk, DACallbackRef callback, DAResponseCallback response, void * responseContext );

extern void DADiskProbe( DADiskRef disk, DACallbackRef callback );

extern void DADiskRefresh( DADiskRef disk, DACallbackRef callback );

extern void DADiskUnmount( DADiskRef disk, DADiskUnmountOptions options, DACallbackRef callback );

extern void DADiskUnmountApprovalCallback( DADiskRef disk, DAResponseCallback response, void * responseContext );

extern void DAIdleCallback( void );

extern void DADiskListCompleteCallback( void );

extern void DAQueueCallback( DACallbackRef callback, DADiskRef argument0, CFTypeRef argument1 );

extern void DAQueueCallbacks( DASessionRef session, _DACallbackKind kind, DADiskRef argument0, CFTypeRef argument1 );

extern void DAQueueReleaseDisk( DADiskRef disk );

extern void DAQueueReleaseSession( DASessionRef session );

extern void DAQueueRequest( DARequestRef request );

extern void DAQueueUnregisterCallback( DACallbackRef callback );

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* !__DISKARBITRATIOND_DAQUEUE__ */
