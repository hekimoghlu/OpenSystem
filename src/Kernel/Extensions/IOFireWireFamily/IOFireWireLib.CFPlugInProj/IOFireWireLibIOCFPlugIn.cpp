/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 11, 2023.
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
#import "IOFireWireLibIOCFPlugIn.h"
#import "IOFireWireLibDevice.h"

namespace IOFireWireLib {
	
	const IOCFPlugInInterface IOCFPlugIn::sInterface = 
	{
		INTERFACEIMP_INTERFACE,
		1, 0, // version/revision
		& IOCFPlugIn::SProbe,
		& IOCFPlugIn::SStart,
		& IOCFPlugIn::SStop
	};
	
	IOCFPlugIn::IOCFPlugIn()
	: IOFireWireIUnknown( reinterpret_cast<const IUnknownVTbl &>( sInterface ) ),
	  mDevice(0)
	{
		// factory counting
		::CFPlugInAddInstanceForFactory( kIOFireWireLibFactoryID );
	}

	IOCFPlugIn::~IOCFPlugIn()
	{
		if (mDevice)
			(**mDevice).Release(mDevice) ;		

		// cleaning up COM bits
		::CFPlugInRemoveInstanceForFactory( kIOFireWireLibFactoryID );
	}
	
	IOReturn
	IOCFPlugIn::Probe( CFDictionaryRef propertyTable, io_service_t service, SInt32 *order )
	{	
		// only load against firewire nubs
		if( !service || !IOObjectConformsTo(service, "IOFireWireNub") )
			return kIOReturnBadArgument;
		
		return kIOReturnSuccess;
	}
	
	IOReturn
	IOCFPlugIn::Start( CFDictionaryRef propertyTable, io_service_t service )
	{
		mDevice = DeviceCOM::Alloc( propertyTable, service ) ;
		if (!mDevice)
			return kIOReturnError ;

		return kIOReturnSuccess ;
	}
	
	IOReturn
	IOCFPlugIn::Stop()
	{
		return kIOReturnSuccess ;
	}
	
	HRESULT
	IOCFPlugIn::QueryInterface( REFIID iid, LPVOID* ppv )
	{
		HRESULT		result = S_OK ;
		*ppv = nil ;
	
		CFUUIDRef	interfaceID	= CFUUIDCreateFromUUIDBytes(kCFAllocatorDefault, iid) ;
	
		if ( CFEqual(interfaceID, IUnknownUUID) ||
			CFEqual(interfaceID, kIOCFPlugInInterfaceID) )
		{
			*ppv = & GetInterface() ;
			AddRef() ;
			::CFRelease(interfaceID) ;
		}
		else 
			// we don't have one of these... let's ask the device interface...
			result = (**mDevice).QueryInterface( mDevice, iid, ppv) ;

		
		return result ;
	}

	IOCFPlugInInterface**
	IOCFPlugIn::Alloc()
	{
		IOCFPlugIn*		me = new IOCFPlugIn ;
		if( !me )
			return nil ;

		return reinterpret_cast<IOCFPlugInInterface **>( & me->GetInterface() );
	}

	IOReturn
	IOCFPlugIn::SProbe(void* self, CFDictionaryRef propertyTable, io_service_t service, SInt32 *order )
	{
		return IOFireWireIUnknown::InterfaceMap<IOCFPlugIn>::GetThis(self)->Probe(propertyTable, service, order) ;
	}
	
	IOReturn
	IOCFPlugIn::SStart(void* self, CFDictionaryRef propertyTable, io_service_t service)
	{
		return IOFireWireIUnknown::InterfaceMap<IOCFPlugIn>::GetThis(self)->Start(propertyTable, service) ;
	}
	
	IOReturn
	IOCFPlugIn::SStop(void* self)
	{
		return IOFireWireIUnknown::InterfaceMap<IOCFPlugIn>::GetThis(self)->Stop() ;
	}
}
