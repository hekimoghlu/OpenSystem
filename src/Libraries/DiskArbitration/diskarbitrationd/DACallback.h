/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 9, 2024.
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
#ifndef __DISKARBITRATIOND_DACALLBACK__
#define __DISKARBITRATIOND_DACALLBACK__

#include <CoreFoundation/CoreFoundation.h>

#include "DADisk.h"
#include "DAInternal.h"
#include "DASession.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

extern DACallbackRef DACallbackCreate( CFAllocatorRef   allocator,
                                       DASessionRef     session,
                                       mach_vm_offset_t address,
                                       mach_vm_offset_t context,
                                       _DACallbackKind  kind,
                                       CFIndex          order,
                                       CFDictionaryRef  match,
                                       CFArrayRef       watch );

extern DACallbackRef    DACallbackCreateCopy( CFAllocatorRef allocator, DACallbackRef callback );
extern mach_vm_offset_t DACallbackGetAddress( DACallbackRef callback );
extern CFTypeRef        DACallbackGetArgument0( DACallbackRef callback );
extern CFTypeRef        DACallbackGetArgument1( DACallbackRef callback );
extern mach_vm_offset_t DACallbackGetContext( DACallbackRef callback );
extern DADiskRef        DACallbackGetDisk( DACallbackRef callback );
extern _DACallbackKind  DACallbackGetKind( DACallbackRef callback );
extern CFDictionaryRef  DACallbackGetMatch( DACallbackRef callback );
extern SInt32           DACallbackGetOrder( DACallbackRef callback );
extern DASessionRef     DACallbackGetSession( DACallbackRef callback );
extern CFAbsoluteTime   DACallbackGetTime( DACallbackRef callback );
extern CFArrayRef       DACallbackGetWatch( DACallbackRef callback );
extern void             DACallbackSetArgument0( DACallbackRef callback, CFTypeRef argument0 );
extern void             DACallbackSetArgument1( DACallbackRef callback, CFTypeRef argument1 );
extern void             DACallbackSetDisk( DACallbackRef callback, DADiskRef disk );
extern void             DACallbackSetMatch( DACallbackRef callback, CFDictionaryRef match );
extern void             DACallbackSetSession( DACallbackRef callback, DASessionRef session );
extern void             DACallbackSetTime( DACallbackRef callback, CFAbsoluteTime time );

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* !__DISKARBITRATIOND_DACALLBACK__ */
