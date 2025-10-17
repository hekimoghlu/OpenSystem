/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 25, 2025.
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
#import <CoreFoundation/CFPlugInCOM.h>


#define INTERFACEIMP_INTERFACE \
	0,	\
	& IOFireWireIUnknown::SQueryInterface,	\
	& IOFireWireIUnknown::SAddRef,	\
	& IOFireWireIUnknown::SRelease

namespace IOFireWireLib {

	class IOFireWireIUnknown: public IUnknown
	{
		protected:
		
			template<class T>
			class InterfaceMap
			{
				public:
					InterfaceMap( const IUnknownVTbl & vTable, T * inObj ): pseudoVTable(vTable), obj(inObj)		{}
					inline static T * GetThis( void* map )		{ return reinterpret_cast<T*>( reinterpret_cast<InterfaceMap*>( map )->obj ) ; }

				private:
					const IUnknownVTbl &	pseudoVTable ;
					T *						obj ;
			} ;
	
		private:
		
			mutable InterfaceMap<IOFireWireIUnknown>	mInterface ;

		protected:

			UInt32										mRefCount ;

		public:
		
			IOFireWireIUnknown( const IUnknownVTbl & interface ) ;
#if IOFIREWIRELIBDEBUG
			virtual ~IOFireWireIUnknown() ;
#else
			virtual ~IOFireWireIUnknown() {}
#endif			
			virtual HRESULT 							QueryInterface( REFIID iid, LPVOID* ppv ) = 0;
			virtual ULONG 								AddRef() ;
			virtual ULONG 								Release() ;
			
			InterfaceMap<IOFireWireIUnknown>&			GetInterface() const		{ return mInterface ; }
			
			static HRESULT STDMETHODCALLTYPE			SQueryInterface(void* self, REFIID iid, LPVOID* ppv) ;
			static ULONG STDMETHODCALLTYPE				SAddRef(void* self) ;
			static ULONG STDMETHODCALLTYPE				SRelease(void* 	self) ;		
	} ;
	
} // namespace
