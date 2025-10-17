/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 29, 2022.
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
 *  IOFireWireLibUnitDirectory.h
 *  IOFireWireLib
 *
 *  Created by NWG on Thu Apr 27 2000.
 *  Copyright (c) 2000 Apple Computer, Inc. All rights reserved.
 *
 */

#import "IOFireWireLibIUnknown.h"
#import "IOFireWireLibPriv.h"

namespace IOFireWireLib {

	class Device ;
	class LocalUnitDirectory: public IOFireWireIUnknown
	{
		typedef ::IOFireWireLocalUnitDirectoryInterface 	Interface ;
		typedef ::IOFireWireLibLocalUnitDirectoryRef		DirRef ;

		protected:
		
			static Interface		sInterface ;		
			UserObjectHandle mKernUnitDirRef ;
			Device&					mUserClient ;
			bool					mPublished ;

			HRESULT					QueryInterface(
											REFIID				iid, 
											void **				ppv ) ;		
		public:
			// --- constructor/destructor ----------
									LocalUnitDirectory( Device& userclient ) ;
			virtual					~LocalUnitDirectory() ;
		
			// --- adding to ROM -------------------
			IOReturn				AddEntry(
											int 				key,
											void*				inBuffer,
											size_t				inLen,
											CFStringRef			inDesc = NULL) ;
			IOReturn				AddEntry(
											int					key,
											UInt32				value,
											CFStringRef			inDesc = NULL) ;
			IOReturn				AddEntry(
											int					key,
											const FWAddress &	value,
											CFStringRef			inDesc = NULL) ;
											
			IOReturn				Publish() ;
			IOReturn				Unpublish() ;

			// --- IUNKNOWN support ----------------
			static Interface**		Alloc( Device& userclient ) ;

			// --- adding to ROM -------------------
			static IOReturn			SAddEntry_Ptr(
											DirRef self,
											int 				key,
											void*				inBuffer,
											size_t				inLen,
											CFStringRef			inDesc = NULL) ;
			static IOReturn			SAddEntry_UInt32(
											DirRef self,
											int					key,
											UInt32				value,
											CFStringRef			inDesc = NULL) ;
			static IOReturn			SAddEntry_FWAddress(
											DirRef self,
											int					key,
											const FWAddress*	value,
											CFStringRef			inDesc = NULL) ;
		
			// Use this function to cause your unit directory to appear in the Mac's config ROM.
			static IOReturn			SPublish( DirRef self ) ;
			static IOReturn			SUnpublish( DirRef self ) ;		
	} ;
}
