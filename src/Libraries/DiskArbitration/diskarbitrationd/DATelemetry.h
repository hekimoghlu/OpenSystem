/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 15, 2022.
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

//
//  DATelemetry.h
//  diskarbitrationd
//
//  Created by Andrew Tran on 8/8/24.
//

#ifndef __DISKARBITRATIOND_DATELEMETRY__
#define __DISKARBITRATIOND_DATELEMETRY__

#include <CoreFoundation/CoreFoundation.h>

int DATelemetrySendProbeEvent      ( int status , CFStringRef fsType , CFStringRef fsImplementation , uint64_t durationNs , int cleanStatus );
int DATelemetrySendFSCKEvent       ( int status , CFStringRef fsType , CFStringRef fsImplementation , uint64_t durationNs , uint64_t volumeSize );
int DATelemetrySendMountEvent      ( int status , CFStringRef fsType , bool useUserFS , uint64_t durationNs );
int DATelemetrySendEjectEvent      ( int status , CFStringRef fsType , pid_t dissenterPid );
int DATelemetrySendTerminationEvent( CFStringRef fsType ,
                                     CFStringRef fsImplementation ,
                                     bool isMounted ,
                                     bool isAppeared ,
                                     bool isProbing ,
                                     bool isFSCKRunning ,
                                     bool isMounting ,
                                     bool isUnrepairable ,
                                     bool isRemoved );
int DATelemetrySendUnmountEvent    ( int status , CFStringRef fsType , CFStringRef fsImplementation ,
                                     bool forced , pid_t dissenterPid ,
                                     bool dissentedViaAPI , uint64_t durationNs );

/* Special telemetry status codes */
#define DA_STATUS_FSTAB_MOUNT_SKIPPED 255
#define DA_STATUS_FSTAB_MOUNT_ADDED   256

#endif /* __DISKARBITRATIOND_DATELEMETRY__ */
