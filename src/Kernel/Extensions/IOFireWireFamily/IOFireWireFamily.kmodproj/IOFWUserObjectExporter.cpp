/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 10, 2025.
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
#undef min
#undef max

#import <sys/systm.h>   // for snprintf...

#import "IOFWUserObjectExporter.h"
#import "FWDebugging.h"

#undef super
#define super OSObject

OSDefineMetaClassAndStructors ( IOFWUserObjectExporter, super );

OSMetaClassDefineReservedUnused(IOFWUserObjectExporter, 0);
OSMetaClassDefineReservedUnused(IOFWUserObjectExporter, 1);
OSMetaClassDefineReservedUnused(IOFWUserObjectExporter, 2);
OSMetaClassDefineReservedUnused(IOFWUserObjectExporter, 3);
OSMetaClassDefineReservedUnused(IOFWUserObjectExporter, 4);
OSMetaClassDefineReservedUnused(IOFWUserObjectExporter, 5);
OSMetaClassDefineReservedUnused(IOFWUserObjectExporter, 6);
OSMetaClassDefineReservedUnused(IOFWUserObjectExporter, 7);

// createWithOwner
//
// static factory method

IOFWUserObjectExporter * IOFWUserObjectExporter::createWithOwner( OSObject * owner )
{
	DebugLog( "IOFWUserObjectExporter::create\n" );

	bool success = true;
	
	IOFWUserObjectExporter * object = NULL;
	
	object = OSTypeAlloc( IOFWUserObjectExporter );
	if( object == NULL )
	{
		success = false;
	}
	
	if( success )
	{
		success = object->initWithOwner( owner );
	}
	
	if( !success )
	{
		if( object )
		{
			object->release();
			object = NULL;
		}
	}
	
	return object;
}

bool
IOFWUserObjectExporter::init()
{
	fLock = IOLockAlloc () ;
	if ( ! fLock )
		return false ;
	
	return super::init () ;
}

bool
IOFWUserObjectExporter::initWithOwner ( OSObject * owner )
{
	fOwner = owner ;
	
	return init() ;
}

void
IOFWUserObjectExporter::free ()
{
	DebugLog( "free object exporter %p, fObjectCount = %d\n", this, fObjectCount ) ;

	removeAllObjects () ;

	if ( fLock ) {
		IOLockFree( fLock ) ;
		fLock = NULL;
	}
	
	fOwner = NULL ;
	
	super::free () ;
}

bool
IOFWUserObjectExporter::serialize ( 
	OSSerialize * s ) const
{
	lock() ;

	const OSString * keys[ 3 ] =
	{
		OSString::withCString( "capacity" )
		, OSString::withCString( "used" )
		, OSString::withCString( "objects" )
	} ;
	
	const OSObject * objects[ 3 ] =
	{
		OSNumber::withNumber( (unsigned long long)fCapacity, 32 )
		, OSNumber::withNumber( (unsigned long long)fObjectCount, 32 )
		, fObjects ? OSArray::withObjects( fObjects, fObjectCount ) : OSArray::withCapacity(0)
	} ;
	
	OSDictionary * dict = OSDictionary::withObjects( objects, keys, sizeof( keys ) / sizeof( OSObject* ) ) ;

	if ( !dict )
	{
		unlock() ;
		return false ;
	}
		
	bool result = dict->serialize( s ) ;
	
	unlock() ;

	return result ;
}

IOReturn
IOFWUserObjectExporter::addObject ( OSObject * obj, CleanupFunction cleanupFunction, IOFireWireLib::UserObjectHandle * outHandle )
{
	IOReturn error = kIOReturnSuccess ;
	
	lock () ;
	
	if ( ! fObjects )
	{
		fCapacity = 8 ;
		fObjects = (const OSObject **) new const OSObject * [ fCapacity ] ;
		fCleanupFunctions = new CleanupFunctionWithExporter[ fCapacity ] ;
		
		if ( ! fObjects || !fCleanupFunctions )
		{
			DebugLog( "Couldn't make fObjects\n" ) ;
			error = kIOReturnNoMemory ;
		}
	}
	
	// if at capacity, expand pool
	if ( fObjectCount == fCapacity )
	{
		unsigned newCapacity = fCapacity + ( fCapacity >> 1 ) ;
		if ( newCapacity > 0xFFFE )
			newCapacity = 0xFFFE ;
			
		if ( newCapacity == fCapacity )	// can't grow!
		{
			DebugLog( "Can't grow object exporter\n" ) ;
			error = kIOReturnNoMemory ;
		}
		
		const OSObject ** newObjects = NULL ;
		CleanupFunctionWithExporter * newCleanupFunctions = NULL ;

		if ( ! error )
		{
			newObjects = (const OSObject **) new OSObject * [ newCapacity ] ;
		
			if ( !newObjects )
				error = kIOReturnNoMemory ;
		}
		
		if ( !error )
		{
			newCleanupFunctions = new CleanupFunctionWithExporter[ newCapacity ] ;
			if ( !newCleanupFunctions )
				error = kIOReturnNoMemory ;
		}
		
		if ( ! error )
		{
			bcopy ( fObjects, newObjects, fCapacity * sizeof ( OSObject * ) ) ;
			delete[] fObjects ;

			bcopy ( fCleanupFunctions, newCleanupFunctions, fCapacity * sizeof( CleanupFunction * ) ) ;
			delete[] fCleanupFunctions ;

			fObjects = newObjects ;
			fCleanupFunctions = newCleanupFunctions ;
			fCapacity = newCapacity ;
		}
	}
	
	if ( ! error )
	{
		error = kIOReturnNoMemory ;
		unsigned index = 0 ;
		
		while ( index < fCapacity )
		{
			if ( ! fObjects [ index ] )
			{
				obj->retain () ;
				fObjects[ index ] = obj ;
				fCleanupFunctions[ index ] = (CleanupFunctionWithExporter)cleanupFunction ;
				*outHandle = (IOFireWireLib::UserObjectHandle)(index + 1) ;		// return index + 1; this means 0 is always an invalid/NULL index...
				++fObjectCount ;
				error = kIOReturnSuccess ;
				break ;
			}
			
			++index ;
		}
	}
	
	unlock () ;

	ErrorLogCond( error, "fExporter->addObject returning error %x\n", error ) ;
	
	return error ;
}

void
IOFWUserObjectExporter::removeObject ( IOFireWireLib::UserObjectHandle handle )
{
	if ( !handle )
	{
		return ;
	}
	
	lock () ;
	
	DebugLog("user object exporter removing handle %d\n", (uint32_t)handle);

	unsigned index = (unsigned)handle - 1 ;		// handle is object's index + 1; this means 0 is always in invalid/NULL index...
	
	const OSObject * object = NULL ;
	CleanupFunctionWithExporter cleanupFunction = NULL ;
	
	if ( fObjects && ( index < fCapacity ) )
	{
		if ( fObjects[ index ] )
		{
			DebugLog( "found object %p (%s), retain count=%d\n", fObjects[ index ], fObjects[ index ]->getMetaClass()->getClassName(), fObjects[ index ]->getRetainCount() );
			
			object = fObjects[ index ] ;
			fObjects[ index ] = NULL ;

			cleanupFunction = fCleanupFunctions[ index ] ;				
			fCleanupFunctions[ index ] = NULL ;
			
			--fObjectCount ;
		}
	}

	unlock () ;

	if ( object )
	{
		if ( cleanupFunction )
		{
			InfoLog("IOFWUserObjectExporter<%p>::removeObject() -- calling cleanup function for object %p of class %s\n", this, object, object->getMetaClass()->getClassName() ) ;
			(*cleanupFunction)( object, this ) ;
		}
		
		object->release() ;
        object= NULL;
	}
	
}

const IOFireWireLib::UserObjectHandle
IOFWUserObjectExporter::lookupHandle ( OSObject * object ) const
{
	IOFireWireLib::UserObjectHandle	out_handle = 0;
	
	if ( !object )
	{
		return 0;
	}
	
	lock () ;

	unsigned index = 0 ;
	
	while ( index < fCapacity )
	{
		if( fObjects[index] == object )
		{
			out_handle = (IOFireWireLib::UserObjectHandle)(index + 1) ;		// return index + 1; this means 0 is always an invalid/NULL index...
			break;
		}
		
		++index;
	}
	
	unlock ();
	
	return out_handle;
}

const OSObject *
IOFWUserObjectExporter::lookupObject ( IOFireWireLib::UserObjectHandle handle ) const
{
	if ( !handle )
	{
		return NULL ;
	}

	const OSObject * result = NULL ;
	
	lock () ;
	
	unsigned index = (unsigned)handle - 1 ;
	
	if ( fObjects && ( index < fCapacity ) )
	{
		result = fObjects [ index ] ;
		if ( result )
		{
			result->retain() ;
		}
	}
		
	unlock () ;
	
	return result ;
}

const OSObject *
IOFWUserObjectExporter::lookupObjectForType( IOFireWireLib::UserObjectHandle handle, const OSMetaClass * toType ) const
{
	if( !handle )
	{
		return NULL;
	}

	const OSObject * result = NULL;
	
	lock ();
	
	unsigned index = (unsigned)handle - 1;
	
	if ( fObjects && ( index < fCapacity ) )
	{
		result = fObjects [ index ];
	}
	
	if( result )
	{
		result = (OSObject*)OSMetaClassBase::safeMetaCast( result, toType );
	}
	
	if( result )
	{
		result->retain();
	}
	
	unlock ();
	
	return result;
}

void
IOFWUserObjectExporter::removeAllObjects ()
{
	lock () ;
	
	const OSObject ** objects = NULL ;
	CleanupFunctionWithExporter * cleanupFunctions = NULL ;

	unsigned capacity = fCapacity ;

	if ( fObjects )
	{		
		objects = IONew( const OSObject *, capacity ) ;
		cleanupFunctions = IONew( CleanupFunctionWithExporter, capacity ) ;
	
		if ( objects )
			bcopy( fObjects, objects, sizeof( const OSObject * ) * capacity ) ;
		
		if ( cleanupFunctions )
			bcopy( fCleanupFunctions, cleanupFunctions, sizeof( CleanupFunction) * capacity ) ;
			
		delete[] fObjects ;
		fObjects = NULL ;
		
		delete[] fCleanupFunctions ;
		fCleanupFunctions = NULL ;		

		fObjectCount = 0 ;
		fCapacity = 0 ;
	}
	
	unlock() ;

	if ( objects && cleanupFunctions )
	{
		for ( unsigned index=0; index < capacity; ++index )
		{
			if ( objects[index] )
			{
				InfoLog("IOFWUserObjectExporter<%p>::removeAllObjects() -- remove object %p of class %s\n", this, objects[ index ], objects[ index ]->getMetaClass()->getClassName() ) ;
				
				if ( cleanupFunctions[ index ] )
				{
					InfoLog("IOFWUserObjectExporter<%p>::removeAllObjects() -- calling cleanup function for object %p of type %s\n", this, objects[ index ], objects[ index ]->getMetaClass()->getClassName() ) ;
					(*cleanupFunctions[ index ])( objects[ index ], this ) ;
				}
				
                if(objects[index]) {//24465629
                    objects[index]->release() ;
                    objects[index]=NULL;
                }
			}
		}

		IODelete( objects, const OSObject *, capacity ) ;
		IODelete( cleanupFunctions, CleanupFunctionWithExporter, capacity ) ;
	}
}

// getOwner
//
//

OSObject *
IOFWUserObjectExporter::getOwner() const
{
	return fOwner;
}

// lock
//
//

void 
IOFWUserObjectExporter::lock( void ) const
{ 
	if ( fLock ) {
		IOLockLock ( fLock ); 
	}
}

// unlock
//
//

void 
IOFWUserObjectExporter::unlock( void ) const
{ 
	if ( fLock ) {
		IOLockUnlock ( fLock ); 
	}
}

