/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 3, 2023.
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
#ifndef _IOKIT_IOFIREWIRESBP2USERCLIENTCOMMON_H_
#define _IOKIT_IOFIREWIRESBP2USERCLIENTCOMMON_H_

#define kIOFireWireSBP2LibConnection 12

enum IOFWSBP2UserClientCommandCodes {
    kIOFWSBP2UserClientOpen,						// kIOUCScalarIScalarO 0,0
    kIOFWSBP2UserClientClose,						// kIOUCScalarIScalarO 0,0
    kIOFWSBP2UserClientCreateLogin,					// kIOUCScalarIScalarO 0,1
    kIOFWSBP2UserClientReleaseLogin,				// kIOUCScalarIScalarO 1,0
    kIOFWSBP2UserClientSubmitLogin,					// kIOUCScalarIScalarO 1,0
    kIOFWSBP2UserClientSubmitLogout,				// kIOUCScalarIScalarO 1,0
    kIOFWSBP2UserClientSetLoginFlags,				// kIOUCScalarIScalarO 2,0
    kIOFWSBP2UserClientGetMaxCommandBlockSize,		// kIOUCScalarIScalarO 1,1
    kIOFWSBP2UserClientGetLoginID,					// kIOUCScalarIScalarO 1,1
    kIOFWSBP2UserClientSetReconnectTime,			// kIOUCScalarIScalarO 1,0
    kIOFWSBP2UserClientSetMaxPayloadSize,			// kIOUCScalarIScalarO 1,0
    kIOFWSBP2UserClientCreateORB,					// kIOUCScalarIScalarO 0,1
    kIOFWSBP2UserClientReleaseORB,					// kIOUCScalarIScalarO 1,0
    kIOFWSBP2UserClientSubmitORB,					// kIOUCScalarIScalarO 1,0
    kIOFWSBP2UserClientSetCommandFlags,  			// kIOUCScalarIScalarO 2,0
    kIOFWSBP2UserClientSetMaxORBPayloadSize, 		// kIOUCScalarIScalarO 2,0
    kIOFWSBP2UserClientSetCommandTimeout, 			// kIOUCScalarIScalarO 2,0
    kIOFWSBP2UserClientSetCommandGeneration, 		// kIOUCScalarIScalarO 2,0
    kIOFWSBP2UserClientSetToDummy,	  				// kIOUCScalarIScalarO 1,0
    kIOFWSBP2UserClientSetCommandBuffersAsRanges,	// kIOUCScalarIScalarO 6,0
    kIOFWSBP2UserClientReleaseCommandBuffers, 		// kIOUCScalarIScalarO 1,0
    kIOFWSBP2UserClientSetCommandBlock,	  			// kIOUCScalarIScalarO 3,0
	kIOFWSBP2UserClientCreateMgmtORB,     			// kIOUCScalarIScalarO 0,1
	kIOFWSBP2UserClientReleaseMgmtORB,   			// kIOUCScalarIScalarO 1,0
	kIOFWSBP2UserClientSubmitMgmtORB,    			// kIOUCScalarIScalarO 1,0
	kIOFWSBP2UserClientMgmtORBSetCommandFunction,   // kIOUCScalarIScalarO 2,0
	kIOFWSBP2UserClientMgmtORBSetManageeORB,  		// kIOUCScalarIScalarO 2,0
	kIOFWSBP2UserClientMgmtORBSetManageeLogin,    	// kIOUCScalarIScalarO 2,0
	kIOFWSBP2UserClientMgmtORBSetResponseBuffer,    // kIOUCScalarIScalarO 3,0
	kIOFWSBP2UserClientLSIWorkaroundSetCommandBuffersAsRanges, // kIOUCScalarIScalarO 6,0
	kIOFWSBP2UserClientMgmtORBLSIWorkaroundSyncBuffersForOutput, // kIOUCScalarIScalarO 1,0
	kIOFWSBP2UserClientMgmtORBLSIWorkaroundSyncBuffersForInput, // kIOUCScalarIScalarO 1,0
    kIOFWSBP2UserClientOpenWithSessionRef,			// kIOUCScalarIScalarO 1,0
	kIOFWSBP2UserClientGetSessionRef,				// kIOUCScalarIScalarO 0,1
	kIOFWSBP2UserClientRingDoorbell,				// kIOUCScalarIScalarO 1, 0
	kIOFWSBP2UserClientEnableUnsolicitedStatus, // kIOUCScalarIScalarO 1, 0
	kIOFWSBP2UserClientSetBusyTimeoutRegisterValue,   // kIOUCScalarIScalarO 2, 0
	kIOFWSBP2UserClientSetORBRefCon, 				// kIOUCScalarIScalarO 2, 0
	kIOFWSBP2UserClientSetPassword,					// kIOUCScalarIScalarO 3, 0
    kIOFWSBP2UserClientSetMessageCallback,   		// kIOUCScalarIScalarO 2, 0
    kIOFWSBP2UserClientSetLoginCallback,			// kIOUCScalarIScalarO 2, 0
    kIOFWSBP2UserClientSetLogoutCallback,			// kIOUCScalarIScalarO 2, 0
    kIOFWSBP2UserClientSetUnsolicitedStatusNotify, 	// kIOUCScalarIScalarO 2, 0
    kIOFWSBP2UserClientSetStatusNotify, 			// kIOUCScalarIScalarO 2, 0
	kIOFWSBP2UserClientSetMgmtORBCallback,  		// kIOUCScalarIScalarO 3, 0
	kIOFWSBP2UserClientSubmitFetchAgentReset,  		// kIOUCScalarIScalarO 3, 0
	kIOFWSBP2UserClientSetFetchAgentWriteCompletion, // kIOUCScalarIScalaO 2, 0
    kIOFWSBP2UserClientNumCommands
};

#endif