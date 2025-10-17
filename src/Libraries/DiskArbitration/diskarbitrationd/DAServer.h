/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 12, 2025.
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
#ifndef __DISKARBITRATIOND_DASERVER__
#define __DISKARBITRATIOND_DASERVER__

#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#include <SystemConfiguration/SystemConfiguration.h>
#include <dispatch/private.h>
#include <dispatch/dispatch.h>

#include "DADisk.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
#if TARGET_OS_OSX
extern void _DAConfigurationCallback( SCDynamicStoreRef store, CFArrayRef keys, void * info );
#endif
#if TARGET_OS_IOS
extern void DARegisterForUnlockNotification( void );
#endif
extern DADiskRef DADiskListGetDisk( const char * diskID );
extern void _DAMediaAppearedCallback( void * context, io_iterator_t notification );
extern void _DAMediaDisappearedCallback( void * context, io_iterator_t notification );
extern void _DAServerCallback( CFMachPortRef port, void * message, CFIndex messageSize, void * info );
extern kern_return_t _DAServerSessionCancel( mach_port_t _session );
extern void _DAVolumeMountedCallback( void );
extern void _DADiskCreateFromFSStat(struct statfs *fs);
extern void _DAVolumeUnmountedCallback( void );
extern void _DAVolumeUpdatedCallback( void );
#if TARGET_OS_OSX
extern void _DAVolumeMountedMachHandler( void *context, dispatch_mach_reason_t reason,
                                     dispatch_mach_msg_t msg, mach_error_t err );
extern void _DAVolumeUnmountedMachHandler( void *context, dispatch_mach_reason_t reason,
                                       dispatch_mach_msg_t msg, mach_error_t err );
extern void _DAVolumeUpdatedMachHandler( void *context, dispatch_mach_reason_t reason,
                                     dispatch_mach_msg_t msg, mach_error_t err );

#endif
extern dispatch_workloop_t DAServerWorkLoop( void );
void DAServerMachHandler( void *context, dispatch_mach_reason_t reason, dispatch_mach_msg_t msg, mach_error_t error );
void DAServerInit ( void );

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* !__DISKARBITRATIOND_DASERVER__ */
