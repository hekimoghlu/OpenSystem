/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 30, 2023.
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
#import "IOFireWireLibIUnknown.h"
#import <assert.h>
#import <string.h>		// bzero

namespace IOFireWireLib {

	IOFireWireIUnknown::IOFireWireIUnknown( const IUnknownVTbl & interface )
	: mInterface( interface, this ),
	  mRefCount(1) 
	{
	}
#if IOFIREWIRELIBDEBUG
	IOFireWireIUnknown::~IOFireWireIUnknown()
	{
	}	
#endif

	// static
	HRESULT
	IOFireWireIUnknown::SQueryInterface(void* self, REFIID iid, void** ppv)
	{
		return IOFireWireIUnknown::InterfaceMap<IOFireWireIUnknown>::GetThis(self)->QueryInterface(iid, ppv) ;
	}
	
	UInt32
	IOFireWireIUnknown::SAddRef(void* self)
	{
		return IOFireWireIUnknown::InterfaceMap<IOFireWireIUnknown>::GetThis(self)->AddRef() ;
	}
	
	ULONG
	IOFireWireIUnknown::SRelease(void* self)
	{
		return IOFireWireIUnknown::InterfaceMap<IOFireWireIUnknown>::GetThis(self)->Release() ;
	}
	
	ULONG
	IOFireWireIUnknown::AddRef()
	{
		return ++mRefCount ;
	}
	
	ULONG
	IOFireWireIUnknown::Release()
	{
		assert( mRefCount > 0) ;
	
		UInt32 newCount = mRefCount;
		
		if (mRefCount == 1)
		{
			mRefCount = 0 ;
			delete this ;
		}
		else
			mRefCount-- ;
		
		return newCount ;
	}
} // namespace
