/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 25, 2023.
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
#ifndef _IOKIT_IO_FIREWIRE_SERIAL_BUS_PROTOCOL_TRANSPORT_TIMESTAMPS_H_
#define _IOKIT_IO_FIREWIRE_SERIAL_BUS_PROTOCOL_TRANSPORT_TIMESTAMPS_H_

#include <IOKit/IOTypes.h>

#include <sys/kdebug.h>
#include <IOKit/scsi/IOSCSIArchitectureModelFamilyTimestamps.h>

#ifdef __cplusplus
extern "C" {
#endif




/* The trace codes consist of the following (see IOSCSIArchitectureModelFamilyTimestamps.h):
 *
 * ----------------------------------------------------------------------
 *|              |               |              |               |Func   |
 *| Class (8)    | SubClass (8)  | SAM Class(6) |  Code (8)     |Qual(2)|
 * ----------------------------------------------------------------------
 *
 * DBG_IOKIT(05h)  DBG_IOSAM(27h)	  (20h)
 *
 * See <sys/kdebug.h> and IOTimeStamp.h for more details.
 *
 */

// FireWire tracepoints								0x05278000 - 0x052783FF
enum
{
	kGUID								= 1,		/* 0x05278004 */
	kLoginRequest						= 2,		/* 0x05278008 */
	kLoginCompletion					= 3,		/* 0x0527800C */
	kLoginLost							= 4,		/* 0x05278010 */
	kLoginResumed						= 5,		/* 0x05278014 */
	kSendSCSICommand1					= 6,		/* 0x05278018 */
	kSendSCSICommand2					= 7,		/* 0x0527801C */
	kSCSICommandSenseData				= 8,		/* 0x05278020 */
	kCompleteSCSICommand				= 9,		/* 0x05278024 */
	kSubmitOrb							= 10,		/* 0x05278028 */
	kStatusNotify						= 11,		/* 0x0527802C */
	kFetchAgentReset					= 12,		/* 0x05278030 */
	kFetchAgentResetComplete			= 13,		/* 0x05278034 */
	kLogicalUnitReset					= 14,		/* 0x05278038 */
	kLogicalUnitResetComplete			= 15		/* 0x0527803C */
};

// Tracepoint macros.
#define FW_TRACE(code)	( ( ( DBG_IOKIT & 0xFF ) << 24) | ( ( DBG_IOSAM & 0xFF ) << 16 ) | ( kSAMClassFireWire << 10 ) | ( ( code & 0xFF ) << 2 ) )

#ifdef __cplusplus
}
#endif


#endif	/* _IOKIT_IO_FIREWIRE_SERIAL_BUS_PROTOCOL_TRANSPORT_TIMESTAMPS_H_ */
