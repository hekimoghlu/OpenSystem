/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 14, 2023.
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
#ifndef _IOKIT_IOFireWireAVCCommand_H
#define _IOKIT_IOFireWireAVCCommand_H
 
#include <IOKit/firewire/IOFWCommand.h>

/*! @class IOFireWireAVCCommand
*/
class IOFireWireAVCCommand : public IOFWCommand
{
    OSDeclareDefaultStructors(IOFireWireAVCCommand)
    
protected:
    IOFWCommand 		*fWriteCmd;
    IOMemoryDescriptor	*fMem;
    const UInt8 		*fCommand;
    
    UInt32 	fCmdLen;
    UInt8 	*fResponse;
    UInt32 	*fResponseLen;
    int		fCurRetries;
    int		fMaxRetries;
    
    UInt32	fWriteGen;
    UInt16	fWriteNodeID;
    bool	bypassRobustCommandResponseMatching;
	
/*! @struct ExpansionData
    @discussion This structure will be used to expand the capablilties of the class in the future.
    */    
    struct ExpansionData { 
		bool 	fStarted;
		bool 	fSyncWakeupSignaled;
	};

/*! @var reserved
    Reserved for future use.  (Internal use only)  */
    ExpansionData *fIOFireWireAVCCommandExpansion;
    
    static void writeDone(void *refcon, IOReturn status, IOFireWireNub *device, IOFWCommand *fwCmd);
    
    virtual IOReturn	complete(IOReturn status);
    virtual IOReturn	execute();
    virtual void		free();
    
public:
    virtual bool init(IOFireWireNub *device, const UInt8 * command, UInt32 cmdLen,
                                                    UInt8 * response, UInt32 * responseLen);
    virtual IOReturn reinit(IOFireWireNub *device, const UInt8 * command, UInt32 cmdLen,
                                                    UInt8 * response, UInt32 * responseLen);
                                                    
    static IOFireWireAVCCommand *withNub(IOFireWireNub *device, const UInt8 * command, UInt32 cmdLen,
                                                    UInt8 * response, UInt32 * responseLen);
                                                    
    static IOFireWireAVCCommand *withNub(IOFireWireNub *device, UInt32 generation,
                const UInt8 * command, UInt32 cmdLen, UInt8 * response, UInt32 * responseLen);
                                                    
    virtual UInt32 handleResponse(UInt16 nodeID, UInt32 len, const void *buf);

    virtual IOReturn resetInterimTimeout();

	virtual UInt32 handleResponseWithSimpleMatching(UInt16 nodeID, UInt32 len, const void *buf);

    virtual IOReturn 	submit(bool queue = false);

private:
    OSMetaClassDeclareReservedUsed(IOFireWireAVCCommand, 0);
    OSMetaClassDeclareReservedUnused(IOFireWireAVCCommand, 1);
    OSMetaClassDeclareReservedUnused(IOFireWireAVCCommand, 2);
    OSMetaClassDeclareReservedUnused(IOFireWireAVCCommand, 3);
};

#endif // _IOKIT_IOFireWireAVCCommand_H

