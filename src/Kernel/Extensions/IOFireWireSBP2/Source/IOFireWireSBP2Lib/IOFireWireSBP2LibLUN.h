/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 14, 2024.
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
#ifndef _IOKIT_IOFIREWIRESBP2LIBLUN_H_
#define _IOKIT_IOFIREWIRESBP2LIBLUN_H_

#include <IOKit/IOCFPlugIn.h>

#include <IOKit/sbp2/IOFireWireSBP2Lib.h>

__BEGIN_DECLS
void *IOFireWireSBP2LibFactory( CFAllocatorRef allocator, CFUUIDRef typeID );
__END_DECLS

class IOFireWireSBP2LibLUN
{

public:

	struct InterfaceMap 
	{
        IUnknownVTbl *pseudoVTable;
        IOFireWireSBP2LibLUN *obj;
    };
	
	IOFireWireSBP2LibLUN( void );
	virtual ~IOFireWireSBP2LibLUN();
	
protected:

	//////////////////////////////////////
	// cf plugin interfaces
	
	static IOCFPlugInInterface 				sIOCFPlugInInterface;
	InterfaceMap 			   				fIOCFPlugInInterface;
	static IOFireWireSBP2LibLUNInterface	sIOFireWireSBP2LibLUNInterface;
	InterfaceMap							fIOFireWireSBP2LibLUNInterface;

	//////////////////////////////////////
	// cf plugin ref counting
	
	CFUUIDRef 	fFactoryId;	
	UInt32 		fRefCount;

	//////////////////////////////////////	
	// user client connection
	
	io_service_t 	fService;
	io_connect_t 	fConnection;

	//////////////////////////////////////	
	// async callbacks
	
	mach_port_t 			fAsyncPort;
	CFMachPortRef			fCFAsyncPort;
	CFRunLoopRef			fCFRunLoop;
	CFRunLoopSourceRef		fCFRunLoopSource;
	IOFWSBP2MessageCallback	fMessageCallbackRoutine;
	void *					fMessageCallbackRefCon;
	IUnknownVTbl **			fLoginInterface;
	
	void *			fRefCon;

	// utility function to get "this" pointer from interface
	static inline IOFireWireSBP2LibLUN *getThis( void *self )
        { return (IOFireWireSBP2LibLUN *) ((InterfaceMap *) self)->obj; };

	//////////////////////////////////////	
	// IUnknown static methods
	
	static HRESULT staticQueryInterface( void * self, REFIID iid, void **ppv );
	virtual HRESULT queryInterface( REFIID iid, void **ppv );

	static UInt32 staticAddRef( void * self );
	virtual UInt32 addRef( void );

	static UInt32 staticRelease( void * self );
	virtual UInt32 release( void );
	
	//////////////////////////////////////
	// CFPlugin static methods
	
	static IOReturn staticProbe( void * self, CFDictionaryRef propertyTable, 
								 io_service_t service, SInt32 *order );
	virtual IOReturn probe( CFDictionaryRef propertyTable, io_service_t service, SInt32 *order );

    static IOReturn staticStart( void * self, CFDictionaryRef propertyTable, 
								 io_service_t service );
    virtual IOReturn start( CFDictionaryRef propertyTable, io_service_t service );

	static IOReturn staticStop( void * self );
	virtual IOReturn stop( void );

	//////////////////////////////////////
	// IOFireWireSBP2LUN static methods
	
	static IOReturn staticOpen( void * self );
	virtual IOReturn open( void );

	static IOReturn staticOpenWithSessionRef( void * self, IOFireWireSessionRef sessionRef );
	virtual IOReturn openWithSessionRef( IOFireWireSessionRef sessionRef );

	static IOFireWireSessionRef staticGetSessionRef(void * self);
	virtual IOFireWireSessionRef getSessionRef( void );

	static void staticClose( void * self );
	virtual void close( void );

	static IOReturn staticAddIODispatcherToRunLoop( void *self, CFRunLoopRef cfRunLoopRef );
	virtual IOReturn addIODispatcherToRunLoop( CFRunLoopRef cfRunLoopRef );

	static void staticRemoveIODispatcherFromRunLoop( void * self );
	virtual void removeIODispatcherFromRunLoop( void );

	static void staticSetMessageCallback( void * self, void * refCon, 
												IOFWSBP2MessageCallback callback );
	virtual void setMessageCallback( void * refCon, IOFWSBP2MessageCallback callback );

	static IUnknownVTbl ** staticCreateLogin( void * self, REFIID iid );
	virtual IUnknownVTbl ** createLogin( REFIID iid );

	static void staticSetRefCon( void * self, void * refCon );
	virtual void setRefCon( void * refCon );

	static void * staticGetRefCon( void * self );
	virtual void * getRefCon( void );

	static IUnknownVTbl ** staticCreateMgmtORB( void * self, REFIID iid );
	virtual IUnknownVTbl ** createMgmtORB( REFIID iid );

	//////////////////////////////////////											
	// callback static methods
	
	static void staticMessageCallback( void *refcon, IOReturn result, 
													io_user_reference_t *args, int numArgs );
	virtual void messageCallback( IOReturn result, io_user_reference_t *args, int numArgs );
	
public:

	static IOCFPlugInInterface **alloc( void );

};

#endif