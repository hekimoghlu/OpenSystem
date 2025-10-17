/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 28, 2022.
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
 *  IOFWUserIsochPortProxy.h
 *  IOFireWireFamily
 *
 *  Created by NWG on Tue Mar 20 2001.
 *  Copyright (c) 2001 Apple Computer, Inc. All rights reserved.
 *
 */

#ifndef _IOKIT_IOFWUserIsochPortProxy_H
#define _IOKIT_IOFWUserIsochPortProxy_H

#import "IOFireWireLibPriv.h"

// public
#import <IOKit/firewire/IOFWLocalIsochPort.h>
#import <IOKit/IOLocks.h>
#import <IOKit/OSMessageNotification.h>

#pragma mark -

class IODCLProgram ;
class IOBufferMemoryDescriptor ;
class IOFireWireUserClient ;
class IOFWDCLPool ;

class IOFWUserLocalIsochPort : public IOFWLocalIsochPort
{
	OSDeclareDefaultStructors( IOFWUserLocalIsochPort )
	
	typedef ::IOFireWireLib::LocalIsochPortAllocateParams AllocateParams ;
	
	protected:
		
		IORecursiveLock*			fLock ;
		mach_vm_address_t			fUserObj ;
		IOFireWireUserClient *		fUserClient ;

		unsigned					fProgramCount ;
		DCLCommand **				fDCLTable ;		// lookup table
		OSAsyncReference64			fStopTokenAsyncRef ;

		UInt8*						fProgramBuffer ; // for old style programs
		IOFWDCLPool *				fDCLPool ;		// for new style programs
		bool						fStarted ;
		
	public:

		// OSObject
		virtual void				free (void) APPLE_KEXT_OVERRIDE;
#if IOFIREWIREDEBUG > 0
		virtual bool				serialize( OSSerialize * s ) const APPLE_KEXT_OVERRIDE;
#endif

		// IOFWLocalIsochPort
		virtual IOReturn			start (void) APPLE_KEXT_OVERRIDE;
		virtual IOReturn			stop (void) APPLE_KEXT_OVERRIDE;

		// me
		bool						initWithUserDCLProgram (	
											AllocateParams * 		params,
											IOFireWireUserClient &	userclient,
											IOFireWireController &	controller ) ;
		IOReturn					importUserProgram (
											IOMemoryDescriptor *	userExportDesc,
											unsigned 				bufferRangeCount, 
											IOAddressRange			userBufferRanges [],
											IOMemoryMap *			bufferMap ) ;
		static	void				s_dclCallProcHandler (
											DCLCallProc * 			dcl ) ;
		IOReturn					setAsyncRef_DCLCallProc ( 
											OSAsyncReference64 		asyncRef ) ;
		IOReturn					modifyJumpDCL ( 
											UInt32 					jumpCompilerData, 
											UInt32 					labelCompilerData ) ;
		IOReturn					modifyDCLSize ( 
											UInt32 					compilerData, 
											IOByteCount 			newSize ) ;	

		inline void					lock ()				{ IORecursiveLockLock ( fLock ) ; }
		inline void					unlock ()			{ IORecursiveLockUnlock ( fLock ) ; }

		IOReturn					convertToKernelDCL ( UserExportDCLUpdateDCLList *pUserExportDCL, DCLUpdateDCLList * dcl ) ;
		IOReturn					convertToKernelDCL ( UserExportDCLJump *pUserExportDCL, DCLJump * dcl ) ;
		IOReturn					convertToKernelDCL ( UserExportDCLCallProc *pUserExportDCL, DCLCallProc * dcl ) ;

		static void					exporterCleanup( const OSObject * self );
		static void					s_nuDCLCallout( void * refcon ) ;
		IOReturn 					userNotify (
											UInt32			notificationType,
											UInt32			numDCLs,
											void *			data,
											IOByteCount		dataSize ) ;
		IOWorkLoop *				createRealtimeThread() ;
} ;

#endif //_IOKIT_IOFWUserIsochPortProxy_H
