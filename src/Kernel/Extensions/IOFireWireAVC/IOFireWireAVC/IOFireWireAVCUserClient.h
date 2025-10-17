/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 12, 2024.
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
#ifndef _IOKIT_IOFIREWIREAVCUSERCLIENT_H
#define _IOKIT_IOFIREWIREAVCUSERCLIENT_H

#include <IOKit/IOUserClient.h>

//#include <IOKit/firewire/IOFireWireController.h>

#include <IOKit/avc/IOFireWireAVCUnit.h>
#include <IOKit/avc/IOFireWireAVCUserClientCommon.h>

#include <IOKit/firewire/IOFWUserObjectExporter.h>

class IOFireWireAVCUserClient;

// A little class to put into the connections set
class IOFireWireAVCConnection : public OSObject
{
    OSDeclareDefaultStructors(IOFireWireAVCConnection)

public:

    UInt32 fPlugAddr;
	UInt32 fChannel;
};

// A wrapper class for async AVC commands created by this user client
class IOFireWireAVCUserClientAsyncCommand : public OSObject
{
    OSDeclareDefaultStructors(IOFireWireAVCUserClientAsyncCommand)
public:
	IOFireWireAVCAsynchronousCommand *pAsyncCommand;
	IOMemoryDescriptor *fMem;
	IOFireWireAVCUserClient *pUserClient;
	UInt32 commandIdentifierHandle;
};


class IOFireWireAVCUserClient : public IOUserClient
{
    OSDeclareDefaultStructors(IOFireWireAVCUserClient)

protected:

    bool					fOpened;
	bool					fStarted;
    task_t					fTask;
	
    IOFireWireAVCNub *		fUnit;
	OSArray *				fConnections;

	IOLock *				fAsyncAVCCmdLock;
	OSArray *				fUCAsyncCommands;
	UInt32					fNextAVCAsyncCommandHandle;
    OSAsyncReference64		fAsyncAVCCmdCallbackInfo;
	
#ifdef kUseAsyncAVCCommandForBlockingAVCCommand
	IOLock *avcCmdLock;
	IOFireWireAVCAsynchronousCommand *pCommandObject;
#endif
    
	IOFireWireLib::UserObjectHandle		fSessionRef;
	
    static void remakeConnections(void *arg);
    virtual IOReturn updateP2PCount(UInt32 addr, SInt32 inc, bool failOnBusReset, UInt32 chan, IOFWSpeed speed);
    virtual IOReturn makeConnection(UInt32 addr, UInt32 chan, IOFWSpeed speed);
    virtual void breakConnection(UInt32 addr);
	virtual IOFireWireAVCUserClientAsyncCommand *FindUCAsyncCommandWithHandle(UInt32 commandHandle);
	
	virtual IOReturn externalMethod( uint32_t selector, 
									IOExternalMethodArguments * arguments, 
									IOExternalMethodDispatch * dispatch, 
									OSObject * target, 
									void * reference);
    
public:
	virtual bool initWithTask(task_t owningTask, void * securityToken, UInt32 type,OSDictionary * properties);
	virtual void free();
    virtual bool start( IOService * provider );
    virtual void stop( IOService * provider );

    virtual IOReturn message(UInt32 type, IOService *provider, void *argument);
    
    virtual IOReturn clientClose( void );
    virtual IOReturn clientDied( void );

    virtual IOReturn open( void *, void *, void *, void *, void *, void * );
	virtual IOReturn openWithSessionRef( IOFireWireLib::UserObjectHandle sessionRef, void *, void *, void *, void *, void * );
	virtual IOReturn getSessionRef( uint64_t * sessionRef, void *, void *, void *, void *, void * );
    virtual IOReturn close( void * = 0, void * = 0, void * = 0, void * = 0, void * = 0, void * = 0);

    virtual IOReturn AVCCommand(UInt8 * cmd, UInt8 * response, UInt32 len, UInt32 *size);
    virtual IOReturn AVCCommandInGen(UInt8 * cmd, UInt8 * response, UInt32 len, UInt32 *size);

    virtual IOReturn updateAVCCommandTimeout( void * = 0, void * = 0, void * = 0, void * = 0, void * = 0, void * = 0);

    virtual IOReturn makeP2PInputConnection( UInt32 plugNo, UInt32 chan, void * = 0, void * = 0, void * = 0, void * = 0);
    virtual IOReturn breakP2PInputConnection( UInt32 plugNo, void * = 0, void * = 0, void * = 0, void * = 0, void * = 0);
    virtual IOReturn makeP2POutputConnection( UInt32 plugNo, UInt32 chan, IOFWSpeed speed, void * = 0, void * = 0, void * = 0);
    virtual IOReturn breakP2POutputConnection( UInt32 plugNo, void * = 0, void * = 0, void * = 0, void * = 0, void * = 0);
	
    virtual IOReturn installUserLibAsyncAVCCommandCallback(io_user_reference_t *asyncRef, uint64_t userRefcon, uint64_t *returnParam);
	
	virtual IOReturn CreateAVCAsyncCommand(UInt8 * cmd, UInt8 * asyncAVCCommandHandle, UInt32 len, UInt32 *refSize);
	virtual IOReturn SubmitAVCAsyncCommand(UInt32 commandHandle);
	virtual IOReturn CancelAVCAsyncCommand(UInt32 commandHandle);
	virtual IOReturn ReleaseAVCAsyncCommand(UInt32 commandHandle);
	virtual void HandleUCAsyncCommandCallback(IOFireWireAVCUserClientAsyncCommand *pUCAsyncCommand);
	virtual IOReturn ReinitAVCAsyncCommand(UInt32 commandHandle, const UInt8 *pCommandBytes, UInt32 len);

	virtual bool requestTerminate( IOService * provider, IOOptionBits options );
};

#endif // _IOKIT_IOFIREWIREAVCUSERCLIENT_H