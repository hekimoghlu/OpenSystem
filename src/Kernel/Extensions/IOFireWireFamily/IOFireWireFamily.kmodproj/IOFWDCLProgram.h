/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 30, 2022.
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
 * Copyright (c) 1999-2002 Apple Computer, Inc.  All rights reserved.
 *
 * HISTORY
 *
 */


#ifndef _IOKIT_IOFWDCLPROGRAM_H
#define _IOKIT_IOFWDCLPROGRAM_H

#include <libkern/c++/OSObject.h>
#include <IOKit/firewire/IOFireWireFamilyCommon.h>
#include <IOKit/firewire/IOFireWireBus.h>
#include <IOKit/IOMemoryCursor.h>
#include <IOKit/IOMapper.h>

/*! @class IODCLProgram
 */
class IODCLProgram : public OSObject
{
	OSDeclareAbstractStructors(IODCLProgram)
	
	private :
	
	void * 						reserved0 ;//fDCLTaskToKernel;
	void * 						reserved1 ;//fDataTaskToKernel;
	void *		 				reserved2 ;//fDataBase;
	void *		 				reserved3 ;//		IOMemoryDescriptor *		fDCLDesc;
	IOMemoryMap *				fBufferMem ;
	void *		 				reserved5 ;//		IOMemoryCursor *			fDataCursor;
	
protected:
	
	/*! @struct ExpansionData
	 @discussion This structure will be used to expand the capablilties of the class in the future.
	 */
	struct ExpansionData
	{
		IOFWIsochResourceFlags		resourceFlags ;
		IODMACommand *              fDMACommand;
	};
	
	/*! @var reserved
	 Reserved for future use.  (Internal use only)  */
	ExpansionData *					fExpansionData ;
	
	public :
	
	virtual void			setIsochResourceFlags ( IOFWIsochResourceFlags flags ) ;	// formerly getPhysicalSegs()
	IOFWIsochResourceFlags	getIsochResourceFlags () const ;
	
protected:
	
	virtual void 			free (void) APPLE_KEXT_OVERRIDE;
	
public:
	
	virtual bool 			init ( IOFireWireBus::DCLTaskInfo * info = NULL ) ;
	virtual IOReturn 		allocateHW (
										IOFWSpeed 			speed,
										UInt32 				chan) = 0;
	virtual IOReturn 		releaseHW () = 0;
	virtual IOReturn 		compile (
									 IOFWSpeed 			speed,
									 UInt32 				chan) = 0;
	virtual IOReturn 		notify (
									IOFWDCLNotificationType		notificationType,
									DCLCommand ** 				dclCommandList,
									UInt32 						numDCLCommands ) = 0;
	virtual IOReturn 		start () = 0;
	virtual void 			stop () = 0;
	virtual IOReturn 		pause ();
	virtual IOReturn 		resume ();
	
	virtual void			setForceStopProc(
											 IOFWIsochChannel::ForceStopNotificationProc proc,
											 void * 						refCon,
											 IOFWIsochChannel *			channel ) ;
	protected :
	
	void					generateBufferMap( DCLCommand * program ) ;
	IOReturn				virtualToPhysical(
											  IOVirtualRange						ranges[],
											  unsigned							rangeCount,
											  IOMemoryCursor::IOPhysicalSegment	outSegments[],
											  unsigned &							outPhysicalSegmentCount,
											  unsigned							maxSegments ) ;
	
	public :
	
	IOMemoryMap *			getBufferMap() const ;
	
	public :
	
	// close/open isoch workloop gate...
	// clients should not need to call these.
	virtual void			closeGate() = 0 ;
	virtual void			openGate() = 0 ;
	
	virtual IOReturn		synchronizeWithIO() = 0 ;
	
	virtual IOMapper *      copyMapper() = 0;
	
private:
	
	OSMetaClassDeclareReservedUsed(IODCLProgram, 0);
	OSMetaClassDeclareReservedUsed(IODCLProgram, 1);
	
	// *** VERIFY: IODCLProgram is now using one of the OSMetaClassDeclareReservedUsed
	OSMetaClassDeclareReservedUsed(IODCLProgram, 2);
	OSMetaClassDeclareReservedUnused(IODCLProgram, 3);
	OSMetaClassDeclareReservedUnused(IODCLProgram, 4);
	
};

#endif /* ! _IOKIT_IOFWDCLPROGRAM_H */




