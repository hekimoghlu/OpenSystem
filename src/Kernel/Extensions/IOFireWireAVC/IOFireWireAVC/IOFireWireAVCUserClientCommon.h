/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 17, 2023.
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
#ifndef _IOKIT_IOFIREWIREAVCUSERCLIENTCOMMON_H_
#define _IOKIT_IOFIREWIREAVCUSERCLIENTCOMMON_H_

#include <IOKit/avc/IOFireWireAVCConsts.h>

#define kIOFireWireAVCLibConnection 13

enum IOFWAVCUserClientCommandCodes {
    kIOFWAVCUserClientOpen,						// kIOUCScalarIScalarO 0,0
    kIOFWAVCUserClientClose,					// kIOUCScalarIScalarO 0,0
    kIOFWAVCUserClientOpenWithSessionRef,		// kIOUCScalarIScalarO 1,0
	kIOFWAVCUserClientGetSessionRef,			// kIOUCScalarIScalarO 0,1
    kIOFWAVCUserClientAVCCommand,				// kIOUCStructIStructO -1,-1
    kIOFWAVCUserClientAVCCommandInGen,			// kIOUCStructIStructO -1,-1
    kIOFWAVCUserClientUpdateAVCCommandTimeout,	// kIOUCScalarIScalarO 0,0
    kIOFWAVCUserClientMakeP2PInputConnection,	// KIOUCScalarIScalarO 1, 0
    kIOFWAVCUserClientBreakP2PInputConnection,	// KIOUCScalarIScalarO 1, 0
    kIOFWAVCUserClientMakeP2POutputConnection,	// KIOUCScalarIScalarO 1, 0
    kIOFWAVCUserClientBreakP2POutputConnection,	// KIOUCScalarIScalarO 1, 0
	kIOFWAVCUserClientCreateAsyncAVCCommand,	// kIOUCStructIStructO -1,-1
	kIOFWAVCUserClientSubmitAsyncAVCCommand,	// kIOUCScalarIScalarO 1, 0
	kIOFWAVCUserClientCancelAsyncAVCCommand,	// kIOUCScalarIScalarO 1, 0
	kIOFWAVCUserClientReleaseAsyncAVCCommand,	// kIOUCScalarIScalarO 1, 0
	kIOFWAVCUserClientReinitAsyncAVCCommand,	// // kIOUCScalarIStructI 1, -1
    kIOFWAVCUserClientNumCommands
};

enum IOFWAVCUserClientAsyncCommandCodes {
    kIOFWAVCUserClientInstallAsyncAVCCommandCallback = kIOFWAVCUserClientNumCommands,	// kIOUCScalarIScalarO 1, 1
    kIOFWAVCUserClientNumAsyncCommands
};


enum IOFWAVCProtocolUserClientCommandCodes {
    kIOFWAVCProtocolUserClientSendAVCResponse,   		// kIOUCScalarIStructI 2, -1
    kIOFWAVCProtocolUserClientFreeInputPlug,			// kIOUCScalarIScalarO 1, 0
    kIOFWAVCProtocolUserClientReadInputPlug,			// kIOUCScalarIScalarO 1, 1
    kIOFWAVCProtocolUserClientUpdateInputPlug,			// kIOUCScalarIScalarO 3, 0
    kIOFWAVCProtocolUserClientFreeOutputPlug,			// kIOUCScalarIScalarO 1, 0
    kIOFWAVCProtocolUserClientReadOutputPlug,			// kIOUCScalarIScalarO 1, 1
    kIOFWAVCProtocolUserClientUpdateOutputPlug,			// kIOUCScalarIScalarO 3, 0
    kIOFWAVCProtocolUserClientReadOutputMasterPlug,		// kIOUCScalarIScalarO 0, 1
    kIOFWAVCProtocolUserClientUpdateOutputMasterPlug,	// kIOUCScalarIScalarO 2, 0
    kIOFWAVCProtocolUserClientReadInputMasterPlug,		// kIOUCScalarIScalarO 0, 1
    kIOFWAVCProtocolUserClientUpdateInputMasterPlug,	// kIOUCScalarIScalarO 2, 0
    kIOFWAVCProtocolUserClientPublishAVCUnitDirectory,	// kIOUCScalarIScalarO 0, 0
	kIOFWAVCProtocolUserClientSetSubunitPlugSignalFormat, // kIOUCScalarIScalarO 4, 0
	kIOFWAVCProtocolUserClientGetSubunitPlugSignalFormat, // kIOUCScalarIScalarO 3, 1
	kIOFWAVCProtocolUserClientConnectTargetPlugs,		// kIOUCStructIStructO
	kIOFWAVCProtocolUserClientDisconnectTargetPlugs,	// kIOUCScalarIScalarO 6, 0
	kIOFWAVCProtocolUserClientGetTargetPlugConnection,	// kIOUCStructIStructO
	kIOFWAVCProtocolUserClientAVCRequestNotHandled,		// kIOUCScalarIStructI 4, -1
    kIOFWAVCProtocolUserClientNumCommands
};

enum IOFWAVCProtocolUserClientAsyncCommandCodes {
    kIOFWAVCProtocolUserClientSetAVCRequestCallback = kIOFWAVCProtocolUserClientNumCommands,   		// kIOUCScalarIScalarO 2, 0
    kIOFWAVCProtocolUserClientAllocateInputPlug,			// kIOUCScalarIScalarO 1, 1
    kIOFWAVCProtocolUserClientAllocateOutputPlug,			// kIOUCScalarIScalarO 1, 1
    kIOFWAVCProtocolUserClientInstallAVCCommandHandler,		// kIOUCScalarIScalarO 4, 0
    kIOFWAVCProtocolUserClientAddSubunit,					// kIOUCScalarIScalarO 5, 1
    kIOFWAVCProtocolUserClientNumAsyncCommands
};

typedef struct _AVCConnectTargetPlugsInParams
{
	UInt32 sourceSubunitTypeAndID;
	IOFWAVCPlugTypes sourcePlugType;
	UInt32 sourcePlugNum;
	UInt32 destSubunitTypeAndID;
	IOFWAVCPlugTypes destPlugType;
	UInt32 destPlugNum;
	Boolean lockConnection;
	Boolean permConnection;
}AVCConnectTargetPlugsInParams;

typedef struct _AVCConnectTargetPlugsOutParams
{
	UInt32 sourcePlugNum;
	UInt32 destPlugNum;
}AVCConnectTargetPlugsOutParams;

typedef struct _AVCGetTargetPlugConnectionInParams
{
	UInt32 subunitTypeAndID;
	IOFWAVCPlugTypes plugType;
	UInt32 plugNum;
}AVCGetTargetPlugConnectionInParams;

typedef struct _AVCGetTargetPlugConnectionOutParams
{
	UInt32 connectedSubunitTypeAndID;
	IOFWAVCPlugTypes connectedPlugType;
	UInt32 connectedPlugNum;
	Boolean lockConnection;
	Boolean permConnection;
}AVCGetTargetPlugConnectionOutParams;

#define kAsyncCmdSharedBufInterimRespOffset 0
#define kAsyncCmdSharedBufFinalRespOffset 512

#endif // _IOKIT_IOFIREWIREAVCUSERCLIENTCOMMON_H_