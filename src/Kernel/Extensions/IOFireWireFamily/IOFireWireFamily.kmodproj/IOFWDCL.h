/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 20, 2025.
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
#import <IOKit/firewire/IOFireWireFamilyCommon.h>

#import <libkern/c++/OSObject.h>
#import <libkern/c++/OSSet.h>
#import <IOKit/IOTypes.h>

class IODCLProgram ;
class OSIterator ;
class IOFireWireLink ;
class IOMemoryMap ;

/*! @class IOFWDCL
*/

class IOFWDCL : public OSObject
{
	OSDeclareAbstractStructors( IOFWDCL ) ;
	
	public:
	
		typedef void (*Callback)( void * refcon ) ;

		enum
		{
			kDynamic					= BIT(1)//kNuDCLDynamic,
			,kUpdateBeforeCallback		= BIT(2)//kNuDCLUpdateBeforeCallback
			,kUser						= BIT(18) // kNuDCLUser
			,kBigEndianUpdates			= BIT(19) // NOTE: Don't change this without making similar change to IOFireWireLib's NuDCL::Export(...)!
		} ;

		class InternalData 
		{
			public:
			
				IOFWDCL *			lastBranch ;
		} ;

	protected:
		
		IOFWDCL*			fBranch ;
		Callback			fCallback ;
		volatile UInt32 *	fTimeStampPtr ;
		UInt32				fRangeCount ;
		IOVirtualRange *	fRanges ;
		OSSet*				fUpdateList ;
		OSIterator *		fUpdateIterator ;
		volatile UInt32 *	fUserStatusPtr ;
		void*				fRefcon ;
		UInt32				fFlags ;
		
		InternalData *		fLoLevel ;

	public:

		//
		// IOFWDCL public API:
		//
		
		virtual bool		initWithRanges ( 
											OSSet * 				updateSet, 
											unsigned 				rangesCount = 0, 
											IOVirtualRange 			ranges [] = NULL ) ;

		void				setBranch( IOFWDCL* branch ) ;
		IOFWDCL*			getBranch()	const ;
		void				setTimeStampPtr ( UInt32* timeStampPtr ) ;
		UInt32*				getTimeStampPtr () const ;
		void				setCallback( Callback callback ) ;
		Callback			getCallback() const ;
		void				setStatusPtr( UInt32* statusPtr ) ;
		UInt32*				getStatusPtr() const ;
		void				setRefcon( void * refcon ) ;
		void *				getRefcon() const ;
		const OSSet*		getUpdateList() const ;
		
		virtual IOReturn	addRange ( IOVirtualRange& range ) ;
		virtual IOReturn	setRanges ( UInt32 numRanges, IOVirtualRange ranges[] ) ;
		virtual UInt32		getRanges( UInt32 maxRanges, IOVirtualRange ranges[] ) const ;
		virtual UInt32		countRanges() ;
		virtual IOReturn	getSpan( IOVirtualRange& result ) const ;
		virtual IOByteCount	getSize() const ;
		IOReturn			appendUpdateList( IOFWDCL* updateDCL ) ;
		IOReturn			setUpdateList( OSSet* updateList ) ;
		void				emptyUpdateList() ; 
		void				setFlags( UInt32 flags ) ;
		UInt32				getFlags() const ;
		

		virtual void		update() = 0 ;

		// OSObject
		
		virtual void		free (void) APPLE_KEXT_OVERRIDE;
		
	public:
		
		//
		// internal use only; please don't use... 
		//
		
		virtual IOReturn				compile( IODCLProgram & , bool & ) = 0 ;
		virtual void					link () = 0 ;

		OSMetaClassDeclareReservedUnused ( IOFWDCL, 4 ) ;		// used to be relink()

	public :
		virtual bool					interrupt( bool &, IOFWDCL * & ) = 0 ;
		virtual void					finalize ( IODCLProgram & ) ;
		virtual IOReturn				importUserDCL (
														UInt8 *				data,
														IOByteCount &		dataSize,
														IOMemoryMap *		bufferMap,
														const OSArray *		dcl ) ;
			
	protected :
	
		friend class IOFWDCLFriend ;
		
	public :
	
		// dump DCL info...
		virtual void					debug(void) ;

	public:
		
		//
		// internal use only; please don't use... 
		//
		
		virtual bool					checkForInterrupt() = 0 ;

    OSMetaClassDeclareReservedUsed ( IOFWDCL, 0 ) ;
    OSMetaClassDeclareReservedUnused ( IOFWDCL, 1 ) ;
    OSMetaClassDeclareReservedUnused ( IOFWDCL, 2 ) ;
    OSMetaClassDeclareReservedUnused ( IOFWDCL, 3 ) ;
//	OSMetaClassDeclareReservedUnused ( ***, 4 ) ;			// used above

} ;

#pragma mark -

/*! @class IOFWReceiveDCL
*/

class IOFWReceiveDCL : public IOFWDCL
{
	OSDeclareAbstractStructors( IOFWReceiveDCL )

	protected :
	
		UInt8		fHeaderBytes ;
		bool		fWait ;
	
	public:

		// me
		virtual bool		initWithParams( 
											OSSet *				updateSet, 
											UInt8				headerBytes, 
											unsigned			rangesCount, 
											IOVirtualRange		ranges [] ) ;	
		IOReturn			setWaitControl( bool wait ) ;

	public :

		// internal use only:
		virtual IOReturn				importUserDCL (
														UInt8 *				data,
														IOByteCount &		dataSize,
														IOMemoryMap *		bufferMap,
														const OSArray *		dcl ) APPLE_KEXT_OVERRIDE;
	
	protected :
	
		virtual void		debug(void) APPLE_KEXT_OVERRIDE;

} ;

#pragma mark -

/*! @class IOFWSendDCL
*/

class IOFWSendDCL : public IOFWDCL
{
	OSDeclareAbstractStructors( IOFWSendDCL )

	protected:
	
		UInt32 * 	fUserHeaderPtr ;			// pointer to 2 quadlets containing isoch header for this packet
		UInt32 *	fUserHeaderMaskPtr ;		// pointer to 2 quadlets; used to mask header quadlets
		IOFWDCL *	fSkipBranchDCL ;
		Callback	fSkipCallback ;
		void *		fSkipRefcon ;
		UInt8		fSync ;
		UInt8		fTag ;

	public:

		// OSObject
		virtual void		free(void) APPLE_KEXT_OVERRIDE;
		
		// IOFWDCL
		virtual IOReturn	addRange ( IOVirtualRange& range ) APPLE_KEXT_OVERRIDE;
		virtual IOReturn	setRanges ( UInt32 numRanges, IOVirtualRange ranges[] ) APPLE_KEXT_OVERRIDE;

		// me
		virtual bool		initWithParams( OSSet * 				updateSet, 
											unsigned 				rangesCount = 0, 
											IOVirtualRange 			ranges [] = NULL,
											UInt8					sync = 0,
											UInt8					tag = 0 ) ;
		
		void				setUserHeaderPtr( UInt32* userHeaderPtr, UInt32 * maskPtr ) ;
		UInt32 *			getUserHeaderPtr() ;
		UInt32 *			getUserHeaderMask() ;
		void				setSkipBranch( IOFWDCL * skipBranchDCL ) ;
		IOFWDCL *			getSkipBranch() const ;
		void				setSkipCallback( Callback callback ) ;
		Callback			getSkipCallback() const ;
		void				setSkipRefcon( void * refcon = 0 ) ;
		void *				getSkipRefcon() const ;										
		void				setSync( UInt8 sync ) ;
		UInt8				getSync() const ;												
		void				setTag( UInt8 tag ) ;											
		UInt8				getTag() const ;

	public :
	
		// internal use only:
		virtual IOReturn				importUserDCL (
														UInt8 *				data,
														IOByteCount &		dataSize,
														IOMemoryMap *		bufferMap,
														const OSArray *		dcl ) APPLE_KEXT_OVERRIDE;
	protected :
	
		virtual void		debug(void) APPLE_KEXT_OVERRIDE;
} ;

#pragma mark -

/*! @class IOFWSkipCycleDCL
*/

class IOFWSkipCycleDCL : public IOFWDCL
{
	OSDeclareAbstractStructors( IOFWSkipCycleDCL )

	public:
	
		virtual bool		init(void) APPLE_KEXT_OVERRIDE;
		
		virtual IOReturn	addRange ( IOVirtualRange& range ) APPLE_KEXT_OVERRIDE;
		virtual IOReturn	setRanges ( UInt32 numRanges, IOVirtualRange ranges[] ) APPLE_KEXT_OVERRIDE;
		virtual IOReturn	getSpan( IOVirtualRange& result ) ;

	protected :
	
		virtual void		debug(void) APPLE_KEXT_OVERRIDE;
} ;
