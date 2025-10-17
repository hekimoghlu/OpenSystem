/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 25, 2023.
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
#import <libkern/c++/OSObject.h>
#import <libkern/c++/OSArray.h>
#import <IOKit/IOTypes.h>

class IOFireWireLink ;
class IOFWDCL ;
class IOFWReceiveDCL ;
class IOFWSendDCL ;
class IOFWSkipCycleDCL ;
class IOFireWireUserClient ;
class IOMemoryDescriptor ;
class IOMemoryMap ;
class OSSet;

/*! @class IOFWDCLPool
	@discussion You should never subclass IOFWDCLPool 
*/

class IOFWDCLPool : public OSObject
{
	OSDeclareAbstractStructors( IOFWDCLPool )

	friend class IOFireWireUserClient ;
	friend class IOFWUserLocalIsochPort ;
	
	protected:
	
		class Expansion*		fReserved ;		// for class expansion

		IOFireWireLink *		fLink ;
		UInt8					fCurrentTag ;
		UInt8					fCurrentSync ;
		OSArray*				fProgram ;
		DCLNuDCLLeader			fLeader ;
	
	public:
		
		// OSObject
		
		virtual void						free(void) APPLE_KEXT_OVERRIDE;
		
		// me
		
		virtual bool	 					initWithLink ( IOFireWireLink& link, UInt32 capacity ) ;
		
		virtual void						setCurrentTagAndSync ( UInt8 tag, UInt8 sync ) ;
		
		virtual IOFWReceiveDCL*				appendReceiveDCL ( 
													OSSet * 				updateSet, 
													UInt8 					headerBytes,
													UInt32					rangesCount,
													IOVirtualRange			ranges[] ) ;
		virtual IOFWSendDCL*				appendSendDCL ( 
													OSSet * 				updateSet, 
													UInt32					rangesCount,
													IOVirtualRange			ranges[] ) ;
		virtual IOFWSkipCycleDCL*			appendSkipCycleDCL () ;
		virtual const OSArray *				getProgramRef () const ;
		
	protected :
	
		IOReturn							importUserProgram (
													IOMemoryDescriptor *	userExportDesc,
													unsigned				bufferRangeCount,
													IOAddressRange			bufferRanges[],
													IOMemoryMap *			bufferMap ) ;
		IOReturn							importUserDCL(
													IOFWDCL *				dcl,
													void * 					importData,
													IOByteCount &			dataSize,
													IOMemoryMap *			bufferMap ) ;
													

	protected :
	
		virtual IOFWReceiveDCL *			allocReceiveDCL () = 0 ;
		virtual IOFWSendDCL *				allocSendDCL () = 0 ;
		virtual IOFWSkipCycleDCL *			allocSkipCycleDCL () = 0 ;

	private :
	
		void								appendDCL( IOFWDCL * dcl ) ;

	public :
	
		DCLCommand *						getProgram() ;
													
    OSMetaClassDeclareReservedUnused ( IOFWDCLPool, 0);
    OSMetaClassDeclareReservedUnused ( IOFWDCLPool, 1);
    OSMetaClassDeclareReservedUnused ( IOFWDCLPool, 2);
    OSMetaClassDeclareReservedUnused ( IOFWDCLPool, 3);
    OSMetaClassDeclareReservedUnused ( IOFWDCLPool, 4);
    OSMetaClassDeclareReservedUnused ( IOFWDCLPool, 5);
    OSMetaClassDeclareReservedUnused ( IOFWDCLPool, 6);
    OSMetaClassDeclareReservedUnused ( IOFWDCLPool, 7);		
} ;
