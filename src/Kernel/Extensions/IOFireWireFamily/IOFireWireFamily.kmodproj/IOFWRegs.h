/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 22, 2023.
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
/*
	File:		IOFWRegs.h

	Copyright:	© 1996-1999 by Apple Computer, Inc., all rights reserved.

*/

#ifndef __IOFWREGS_H__
#define __IOFWREGS_H__

#include <IOKit/firewire/IOFireWireFamilyCommon.h>

#ifndef NEW_ERROR_CODES

//	 
// (!)	The following error codes are obsolete...
//		Please use the error codes defined in IOFireWireFamilyCommon.h
//

enum {
	inUseErr					= -4160,				// Item already in use
	notFoundErr					= -4161,				// Item not found
	timeoutErr					= -4162,				// Something timed out
	busReconfiguredErr			= -4163,				// Bus was reconfigured
	invalidIDTypeErr			= -4166,				// Given ID is of an invalid type for the requested operation.
	alreadyRegisteredErr		= -4168,				// Item has already been registered.
	disconnectedErr				= -4169,				// Target of request has been disconnected.
	retryExceededErr			= -4170,				// Retry limit was exceeded.
	addressRangeErr				= -4171,				// Address is not in range.
	addressAlignmentErr			= -4172,				// Address is not of proper alignment.

	multipleTalkerErr			= -4180,				// Tried to install multiple talkers
	channelActiveErr			= -4181,				// Operation not permitted when channel is active
	noListenerOrTalkerErr		= -4182,				// Every isochronous channel must have one talker and at least
														// one listener
	noChannelsAvailableErr		= -4183,				// No supported isochronous channels are available
	channelNotAvailableErr		= -4184,				// Required channel was not available.
	invalidIsochPortIDErr		= -4185,				// An isochronous port ID is invalid.
	invalidFWReferenceTypeErr	= -4186,				// Operation does not support type of given reference ID
	separateBusErr				= -4187,				// Two or more entities are on two or more busses and cannot be associated with eachother.
	badSelfIDsErr				= -4188,				// Received self IDs are bad.

//zzz Do we own these next ID numbers?
	cableVoltageTooLowErr		= -4190,				// Cable power below device's minimum voltage
	cablePowerInsufficientErr	= -4191					// Can't grant power request at this time
};


#endif //NEW_ERROR_CODES

#endif /* __IOFWREGS_H */

