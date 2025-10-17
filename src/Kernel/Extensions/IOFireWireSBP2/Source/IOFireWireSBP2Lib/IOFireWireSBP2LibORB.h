/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 2, 2024.
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
#ifndef _IOKIT_IOFIREWIRESBP2LIBORB_H_
#define _IOKIT_IOFIREWIRESBP2LIBORB_H_

#include "IOFireWireSBP2LibLUN.h"
#include "IOFireWireSBP2UserClientCommon.h"

class IOFireWireSBP2LibORB
{

public:
	
	struct InterfaceMap 
	{
        IUnknownVTbl *pseudoVTable;
        IOFireWireSBP2LibORB *obj;
    };
	
	IOFireWireSBP2LibORB( void );
	virtual ~IOFireWireSBP2LibORB();
	
	virtual IOReturn init( io_connect_t connection, mach_port_t asyncPort );
	
protected:

	typedef struct 
	{
		uint64_t address;
		uint64_t length;
	} FWSBP2PrivateVirtualRange;

	//////////////////////////////////////
	// cf plugin interfaces

	static IOFireWireSBP2LibORBInterface		sIOFireWireSBP2LibORBInterface;
	InterfaceMap								fIOFireWireSBP2LibORBInterface;

	//////////////////////////////////////
	// cf plugin ref counting
	
	UInt32 		fRefCount;
	
	//////////////////////////////////////
	// user client connection
	
	io_connect_t 	fConnection;	// connection to user client in kernel
	mach_port_t 	fAsyncPort;		// async port for callback from kernel
	uint64_t 		fORBRef;  		// reference to kernel orb object

	void *			fRefCon;

	FWSBP2PrivateVirtualRange	*	fRangeScratch;
	UInt32							fRangeScratchLength;
	
	//////////////////////////////////////	
	// IUnknown static methods
	
	static HRESULT staticQueryInterface( void * self, REFIID iid, void **ppv );
	virtual HRESULT queryInterface( REFIID iid, void **ppv );

	static UInt32 staticAddRef( void * self );
	virtual UInt32 addRef( void );

	static UInt32 staticRelease( void * self );
	virtual UInt32 release( void );
	
	//////////////////////////////////////	
	// IOFireWireSBP2LibORB static methods

	static void staticSetRefCon( void * self, void * refCon );
	virtual void setRefCon( void * refCon );

	static void * staticGetRefCon( void * self );
	virtual void * getRefCon( void );

	static void staticSetCommandFlags( void * self, UInt32 flags );
	virtual void setCommandFlags( UInt32 flags );

	static void staticSetMaxORBPayloadSize( void * self, UInt32 size );
	virtual void setMaxORBPayloadSize( UInt32 size );

	static void staticSetCommandTimeout( void * self, UInt32 timeout );
	virtual void setCommandTimeout( UInt32 timeout );

	static void staticSetCommandGeneration( void * self, UInt32 generation );
	virtual void setCommandGeneration( UInt32 generation );

	static void staticSetToDummy( void * self );
	virtual void setToDummy( void );

    static IOReturn staticSetCommandBuffersAsRanges( void * self, FWSBP2VirtualRange * ranges, 
											UInt32 withCount, UInt32 withDirection, 
											UInt32 offset, UInt32 length );
    virtual IOReturn setCommandBuffersAsRanges( FWSBP2VirtualRange * ranges, UInt32 withCount,
											UInt32 withDirection, UInt32 offset, 
											UInt32 length );

    static IOReturn staticReleaseCommandBuffers( void * self );
    virtual IOReturn releaseCommandBuffers( void );

    static IOReturn staticSetCommandBlock( void * self, void * buffer, UInt32 length );
    virtual IOReturn setCommandBlock( void * buffer, UInt32 length );

    static IOReturn staticLSIWorkaroundSetCommandBuffersAsRanges
							( void * self, FWSBP2VirtualRange * ranges, UInt32 withCount,
									UInt32 withDirection, UInt32 offset, UInt32 length );
    virtual IOReturn LSIWorkaroundSetCommandBuffersAsRanges
								( FWSBP2VirtualRange * ranges, UInt32 withCount,
									UInt32 withDirection, UInt32 offset, UInt32 length );

	static IOReturn staticLSIWorkaroundSyncBuffersForOutput( void * self );
	virtual IOReturn LSIWorkaroundSyncBuffersForOutput( void );

    static IOReturn staticLSIWorkaroundSyncBuffersForInput( void * self );
    virtual IOReturn LSIWorkaroundSyncBuffersForInput( void );

public:

	// utility function to get "this" pointer from interface
	static inline IOFireWireSBP2LibORB *getThis( void *self )
        { return (IOFireWireSBP2LibORB *) ((InterfaceMap *) self)->obj; };

	static IUnknownVTbl **alloc( io_connect_t connection, mach_port_t asyncPort );

	virtual UInt32 getORBRef( void );
	
};

#endif