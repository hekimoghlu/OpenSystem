/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 25, 2025.
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
#ifndef _IOKIT_IO_ATAPI_PROTOCOL_TRANSPORT_TIMESTAMPS_H_
#define _IOKIT_IO_ATAPI_PROTOCOL_TRANSPORT_TIMESTAMPS_H_

#include <IOKit/IOTypes.h>

#include <sys/kdebug.h>
#include <IOKit/scsi/IOSCSIArchitectureModelFamilyTimestamps.h>

#ifdef __cplusplus
extern "C" {
#endif


extern UInt32	gATAPIDebugFlags;


/* The trace codes consist of the following (see IOSCSIArchitectureModelFamilyTimestamps.h):
 *
 * ----------------------------------------------------------------------
 *|              |               |              |               |Func   |
 *| Class (8)    | SubClass (8)  | SAM Class(6) |  Code (8)     |Qual(2)|
 * ----------------------------------------------------------------------
 *
 * DBG_IOKIT(05h)  DBG_IOSAM(27h)	  (24h)
 *
 * See <sys/kdebug.h> and IOTimeStamp.h for more details.
 *
 */

// ATAPI tracepoints					0x05279000 - 0x052790FF
enum
{
	kATADeviceInfo				= 1,		/* 0x05279004 */
	kATASendSCSICommand			= 2,		/* 0x05279008 */
	kATASendSCSICommandFailed	= 3,		/* 0x0527900C */
	kATACompleteSCSICommand		= 4,		/* 0x05279010 */
	kATAAbort					= 5,		/* 0x05279014 */
	kATAReset					= 6,		/* 0x05279018 */
	kATAResetComplete			= 7, 		/* 0x0527901C */
	kATAHandlePowerOn			= 8,		/* 0x05279020 */
	kATAPowerOnReset			= 9,		/* 0x05279024 */
	kATAPowerOnNoReset			= 10,		/* 0x05279028 */
	kATAHandlePowerOff			= 11,		/* 0x0527902C */
	kATADriverPowerOff			= 12, 		/* 0x05279030 */
	kATAStartStatusPolling		= 13,		/* 0x05279034 */
	kATAStatusPoll				= 14,		/* 0x05279038 */
	kATAStopStatusPolling		= 15,		/* 0x0527903C */
	kATASendATASleepCmd			= 16,		/* 0x05279040 */
};

// Tracepoint macros.
#define ATAPI_TRACE(code)	( ( ( DBG_IOKIT & 0xFF ) << 24) | ( ( DBG_IOSAM & 0xFF ) << 16 ) | ( kSAMClassATAPI << 10 ) | ( ( code & 0xFF ) << 2 ) )

#ifdef __cplusplus
}
#endif


#endif	/* _IOKIT_IO_ATAPI_PROTOCOL_TRANSPORT_TIMESTAMPS_H_ */
