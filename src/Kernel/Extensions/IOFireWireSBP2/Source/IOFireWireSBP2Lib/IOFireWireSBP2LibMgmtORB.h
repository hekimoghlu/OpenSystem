/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 5, 2025.
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
#ifndef _IOKIT_IOFIREWIRESBP2LIBMGMTORB_H_
#define _IOKIT_IOFIREWIRESBP2LIBMGMTORB_H_

#include "IOFireWireSBP2LibLUN.h"
#include "IOFireWireSBP2UserClientCommon.h"

class IOFireWireSBP2LibMgmtORB
{

public:

	struct InterfaceMap 
	{
        IUnknownVTbl *pseudoVTable;
        IOFireWireSBP2LibMgmtORB *obj;
    };
	
	IOFireWireSBP2LibMgmtORB( void );
	virtual ~IOFireWireSBP2LibMgmtORB();
	
	virtual IOReturn init( io_connect_t connection, mach_port_t asyncPort );
	
protected:

	//////////////////////////////////////
	// cf plugin interfaces
	
	static IOFireWireSBP2LibMgmtORBInterface	sIOFireWireSBP2LibMgmtORBInterface;
	InterfaceMap								fIOFireWireSBP2LibMgmtORBInterface;

	//////////////////////////////////////
	// cf plugin ref counting
	
	UInt32 			fRefCount;
	
	//////////////////////////////////////
	// user client connection
	
	io_connect_t 	fConnection;	// connection to user client in kernel
	mach_port_t 	fAsyncPort;		// async port for callback from kernel
	uint64_t 		fMgmtORBRef;  	// reference to kernel orb object

	IOFWSBP2ORBAppendCallback		fORBCallbackRoutine;
	void *							fORBCallbackRefCon;

	void *			fRefCon;
	
	// utility function to get "this" pointer from interface
	static inline IOFireWireSBP2LibMgmtORB *getThis( void *self )
        { return (IOFireWireSBP2LibMgmtORB *) ((InterfaceMap *) self)->obj; };

	//////////////////////////////////////	
	// IUnknown static methods
	
	static HRESULT staticQueryInterface( void * self, REFIID iid, void **ppv );
	virtual HRESULT queryInterface( REFIID iid, void **ppv );

	static UInt32 staticAddRef( void * self );
	virtual UInt32 addRef( void );

	static UInt32 staticRelease( void * self );
	virtual UInt32 release( void );

	//////////////////////////////////////	
	// IOFireWireSBP2LibMgmtORB static methods
	static IOReturn staticSubmitORB( void * self );
	virtual IOReturn submitORB( void );

	static void staticSetORBCallback( void * self, void * refCon, 
												IOFWSBP2ORBAppendCallback callback );
	virtual void setORBCallback( void * refCon, IOFWSBP2ORBAppendCallback callback );

	static void staticSetRefCon( void * self, void * refCon );
	virtual void setRefCon( void * refCon );

	static void * staticGetRefCon( void * self );
	virtual void * getRefCon( void );

    static IOReturn staticSetCommandFunction( void * self, UInt32 function );
    virtual IOReturn setCommandFunction( UInt32 function );

	static IOReturn staticSetManageeORB( void * self, void * orb );
	virtual IOReturn setManageeORB( void * orb );

	static IOReturn staticSetManageeLogin( void * self, void * login );
	virtual IOReturn setManageeLogin( void * login );

	static IOReturn staticSetResponseBuffer( void * self, void * buf, UInt32 len );
	virtual IOReturn setResponseBuffer( void * buf, UInt32 len );

	//////////////////////////////////////
	// callback static methods
	
	static void staticORBCompletion( void *refcon, IOReturn result, io_user_reference_t *args, int numArgs );
	virtual void ORBCompletion( IOReturn result, io_user_reference_t *args, int numArgs );

public:
	
	static IUnknownVTbl **alloc( io_connect_t connection, mach_port_t asyncPort );

};
#endif
