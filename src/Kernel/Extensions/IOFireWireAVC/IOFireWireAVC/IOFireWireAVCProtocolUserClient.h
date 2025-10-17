/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 19, 2024.
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
#ifndef _IOKIT_IOFIREWIREAVCPROTOCOLUSERCLIENT_H
#define _IOKIT_IOFIREWIREAVCPROTOCOLUSERCLIENT_H

#include <IOKit/IOUserClient.h>
#include <IOKit/firewire/IOFWAddressSpace.h>
#include <IOKit/avc/IOFireWireAVCUserClientCommon.h>
#include <IOKit/avc/IOFireWireAVCTargetSpace.h>

//#include <IOKit/firewire/IOFireWireController.h>
class IOFireWireNub;
class IOFireWirePCRSpace;
class IOFireWireAVCTargetSpace;

class IOFireWireAVCProtocolUserClient : public IOUserClient
{
    OSDeclareDefaultStructors(IOFireWireAVCProtocolUserClient)

protected:
    task_t						fTask;
	bool						fStarted;
    IOFireWireNub *				fDevice;
    IOFireWireBus *				fBus;
    IOFireWirePCRSpace *		fPCRSpace;
    IOFireWireAVCTargetSpace *	fAVCTargetSpace;
    OSSet *						fInputPlugs;
    OSSet *						fOutputPlugs;
    
    static void forwardPlugWrite(void *refcon, UInt16 nodeID, UInt32 plug, UInt32 oldVal, UInt32 newVal);

	static void avcTargetCommandHandler(const AVCCommandHandlerInfo *pCmdInfo,
									 UInt32 generation,
									 UInt16 nodeID,
									 const void *command,
									 UInt32 cmdLen,
									 IOFWSpeed &speed,
									 UInt32 handlerSearchIndex);

	static void avcSubunitPlugHandler(const AVCSubunitInfo *pSubunitInfo,
								   IOFWAVCSubunitPlugMessages plugMessage,
								   IOFWAVCPlugTypes plugType,
								   UInt32 plugNum,
								   UInt32 messageParams,
								   UInt32 generation,
								   UInt16 nodeID);

    virtual IOReturn sendAVCResponse(UInt32 generation, UInt16 nodeID, const char *buffer, UInt32 size);
    virtual IOReturn allocateInputPlug(io_user_reference_t *asyncRef, uint64_t userRefcon, uint64_t *plug);
    virtual IOReturn freeInputPlug(UInt32 plug);
    virtual IOReturn readInputPlug(UInt32 plug, uint64_t *val);
    virtual IOReturn updateInputPlug(UInt32 plug, UInt32 oldVal, UInt32 newVal);
    virtual IOReturn allocateOutputPlug(io_user_reference_t *asyncRef, uint64_t userRefcon, uint64_t *plug);
    virtual IOReturn freeOutputPlug(UInt32 plug);
    virtual IOReturn readOutputPlug(UInt32 plug, uint64_t *val);
    virtual IOReturn updateOutputPlug(UInt32 plug, UInt32 oldVal, UInt32 newVal);
    virtual IOReturn readOutputMasterPlug(uint64_t *val);
    virtual IOReturn updateOutputMasterPlug(UInt32 oldVal, UInt32 newVal);
    virtual IOReturn readInputMasterPlug(uint64_t *val);
    virtual IOReturn updateInputMasterPlug(UInt32 oldVal, UInt32 newVal);
    virtual IOReturn publishAVCUnitDirectory(void);
	virtual IOReturn installAVCCommandHandler(io_user_reference_t *asyncRef, uint64_t subUnitTypeAndID, uint64_t opCode, uint64_t callback, uint64_t refCon);
    virtual IOReturn addSubunit(io_user_reference_t *asyncRef,
								uint64_t subunitType,
								uint64_t numSourcePlugs,
								uint64_t numDestPlugs,
								uint64_t callBack,
								uint64_t refCon,
								uint64_t *subUnitTypeAndID);
	virtual IOReturn setSubunitPlugSignalFormat(UInt32 subunitTypeAndID,
											 IOFWAVCPlugTypes plugType,
											 UInt32 plugNum,
											 UInt32 signalFormat);

	virtual IOReturn getSubunitPlugSignalFormat(UInt32 subunitTypeAndID,
											 IOFWAVCPlugTypes plugType,
											 UInt32 plugNum,
											 uint64_t *pSignalFormat);

	virtual IOReturn connectTargetPlugs(AVCConnectTargetPlugsInParams *inParams,
									 AVCConnectTargetPlugsOutParams *outParams);

	virtual IOReturn disconnectTargetPlugs(UInt32 sourceSubunitTypeAndID,
								   IOFWAVCPlugTypes sourcePlugType,
								   UInt32 sourcePlugNum,
								   UInt32 destSubunitTypeAndID,
								   IOFWAVCPlugTypes destPlugType,
								   UInt32 destPlugNum);

	virtual IOReturn getTargetPlugConnection(AVCGetTargetPlugConnectionInParams *inParams,
										  AVCGetTargetPlugConnectionOutParams *outParams);

    virtual IOReturn AVCRequestNotHandled(UInt32 generation,
										  UInt16 nodeID,
										  IOFWSpeed speed,
										  UInt32 handlerSearchIndex,
										  const char *pCmdBuf,
										  UInt32 cmdLen);

	virtual IOReturn externalMethod( uint32_t selector, 
									IOExternalMethodArguments * arguments, 
									IOExternalMethodDispatch * dispatch, 
									OSObject * target, 
									void * reference);
	
    virtual void free();
    
public:
    virtual bool start( IOService * provider );
    virtual IOReturn newUserClient( task_t owningTask, void * securityID,
                                    UInt32 type, OSDictionary * properties,
                                    IOUserClient ** handler );
    virtual IOReturn clientClose( void );
    virtual IOReturn clientDied( void );

    // Make it easy to find
    virtual bool matchPropertyTable(OSDictionary * table);

};

#endif // _IOKIT_IOFIREWIREAVCPROTOCOLUSERCLIENT_H

