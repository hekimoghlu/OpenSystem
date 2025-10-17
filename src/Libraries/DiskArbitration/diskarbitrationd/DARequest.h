/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 16, 2021.
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
#ifndef __DISKARBITRATIOND_DAREQUEST__
#define __DISKARBITRATIOND_DAREQUEST__

#include <CoreFoundation/CoreFoundation.h>

#include "DADisk.h"
#include "DADissenter.h"
#include "DAInternal.h"
#include "DASession.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

typedef struct __DARequest * DARequestRef;

enum
{
///w:start
    _kDARequestStateMountArgumentNoWrite = 0x08000000,
    _kDARequestStateStagedAuthorize      = 0x00200000,
///w:stop
    kDARequestStateStagedProbe   = 0x00010000,
    kDARequestStateStagedApprove = 0x00100000
};

typedef UInt32 DARequestState;

extern DARequestRef DARequestCreate( CFAllocatorRef allocator,
                                     _DARequestKind kind,
                                     DADiskRef      argument0,
                                     CFIndex        argument1,
                                     CFTypeRef      argument2,
                                     CFTypeRef      argument3,
                                     uid_t          userUID,
                                     gid_t          userGID,
                                     DACallbackRef  callback );

extern Boolean DARequestDispatch( DARequestRef request );

extern void           DARequestDispatchCallback( DARequestRef request, DAReturn status );
extern CFIndex        DARequestGetArgument1( DARequestRef request );
extern CFTypeRef      DARequestGetArgument2( DARequestRef request );
extern CFTypeRef      DARequestGetArgument3( DARequestRef request );
extern DACallbackRef  DARequestGetCallback( DARequestRef request );
extern DADiskRef      DARequestGetDisk( DARequestRef request );
extern DADissenterRef DARequestGetDissenter( DARequestRef request );
extern _DARequestKind DARequestGetKind( DARequestRef request );
extern CFArrayRef     DARequestGetLink( DARequestRef request );
extern Boolean        DARequestGetState( DARequestRef request, DARequestState state );
extern gid_t          DARequestGetUserGID( DARequestRef request );
extern uid_t          DARequestGetUserUID( DARequestRef request );
extern void           DARequestSetCallback( DARequestRef request, DACallbackRef callback );
extern void           DARequestSetDissenter( DARequestRef request, DADissenterRef dissenter );
extern void           DARequestSetLink( DARequestRef request, CFArrayRef link );
extern void           DARequestSetState( DARequestRef request, DARequestState state, Boolean value );
extern void           DARequestSetArgument2( DARequestRef request, CFTypeRef argument2);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* !__DISKARBITRATIOND_DAREQUEST__ */
