/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 28, 2022.
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
/*! @class IOFWUserObjectExporter
	@discussion An IOFWUserObjectExporter is for internal use only. You should never subclass IOFWUserObjectExporter 
*/

	namespace IOFireWireLib
	{
		typedef UInt32      UserObjectHandle;
	}

#ifdef KERNEL

	class IOFWUserObjectExporter : public OSObject
	{
		OSDeclareDefaultStructors (IOFWUserObjectExporter )

		public :
		
			typedef void (*CleanupFunction)( const OSObject * obj );
			typedef void (*CleanupFunctionWithExporter)( const OSObject * obj, IOFWUserObjectExporter * );
			
		private :
		
			unsigned							fCapacity;
			unsigned							fObjectCount;
			const OSObject **					fObjects;
			CleanupFunctionWithExporter *		fCleanupFunctions;
			IOLock *							fLock;
			OSObject *							fOwner;
			
		public :
		
			static IOFWUserObjectExporter *		createWithOwner( OSObject * owner );
			bool								initWithOwner( OSObject * owner );

			virtual bool			init(void) APPLE_KEXT_OVERRIDE;
	
			virtual void			free (void) APPLE_KEXT_OVERRIDE;
			virtual bool			serialize ( OSSerialize * s ) const APPLE_KEXT_OVERRIDE;
			
			// me
			IOReturn				addObject ( OSObject * obj, CleanupFunction cleanup, IOFireWireLib::UserObjectHandle * outHandle );
			void					removeObject ( IOFireWireLib::UserObjectHandle handle );
			
			// the returned object is retained! This is for thread safety.. if someone else released
			// the object from the pool after you got it, you be in for Trouble
			// Release the returned value when you're done!!
			const OSObject *		lookupObject ( IOFireWireLib::UserObjectHandle handle ) const;
			const OSObject *		lookupObjectForType( IOFireWireLib::UserObjectHandle handle, const OSMetaClass * toType ) const;
			void					removeAllObjects ();

			void					lock () const;
			void					unlock () const;
			
			OSObject *				getOwner() const;
		
			// *** WARNING: (when building SBP2) 'const' type qualifier on return type has no effect
			const IOFireWireLib::UserObjectHandle lookupHandle ( OSObject * object ) const;
		
			// don't subclass, but just in case someone does...
			
		private:
		
			OSMetaClassDeclareReservedUnused(IOFWUserObjectExporter, 0);
			OSMetaClassDeclareReservedUnused(IOFWUserObjectExporter, 1);
			OSMetaClassDeclareReservedUnused(IOFWUserObjectExporter, 2);
			OSMetaClassDeclareReservedUnused(IOFWUserObjectExporter, 3);
			OSMetaClassDeclareReservedUnused(IOFWUserObjectExporter, 4);
			OSMetaClassDeclareReservedUnused(IOFWUserObjectExporter, 5);
			OSMetaClassDeclareReservedUnused(IOFWUserObjectExporter, 6);
			OSMetaClassDeclareReservedUnused(IOFWUserObjectExporter, 7);
				
	};

#endif
