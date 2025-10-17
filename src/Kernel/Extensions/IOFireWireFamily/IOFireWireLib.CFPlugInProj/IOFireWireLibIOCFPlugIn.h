/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 5, 2025.
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
#import "IOFireWireLibPriv.h"

#import <IOKit/IOCFPlugIn.h>

namespace IOFireWireLib {

	class IOCFPlugIn: public IOFireWireIUnknown
	{
		private:

			static const IOCFPlugInInterface	sInterface ;
			IOFireWireLibDeviceRef				mDevice ;

		public:

			IOCFPlugIn() ;
			virtual ~IOCFPlugIn() ;
	
			virtual HRESULT					QueryInterface( REFIID iid, LPVOID* ppv ) ;
			static IOCFPlugInInterface**	Alloc() ;
			
		private:
		
			// IOCFPlugin methods
			IOReturn 				Probe(CFDictionaryRef propertyTable, io_service_t service, SInt32 *order );
			IOReturn 				Start(CFDictionaryRef propertyTable, io_service_t service );
			IOReturn		 		Stop();	

			//
			// --- CFPlugin static methods ---------
			//
			
			static IOReturn 		SProbe( void* self, CFDictionaryRef propertyTable, io_service_t service, SInt32 *order );
			static IOReturn 		SStart( void* self, CFDictionaryRef propertyTable, io_service_t service );
			static IOReturn 		SStop( void* self );	
	} ;
	
} // namespace
