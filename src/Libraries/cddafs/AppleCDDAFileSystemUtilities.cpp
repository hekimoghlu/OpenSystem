/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 13, 2022.
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
// AppleCDDAFileSystemUtilities.cpp created by CJS on Fri 19-May-2000

// Project Includes
#ifndef __APPLE_CDDA_FS_UTILS_H__
#include "AppleCDDAFileSystemUtils.h"
#endif

#ifndef __APPLE_CDDA_FS_DEBUG_H__
#include "AppleCDDAFileSystemDebug.h"
#endif

// System Includes
#include <libkern/c++/OSData.h>
#include <libkern/c++/OSObject.h>
#include <libkern/c++/OSIterator.h>
#include <libkern/c++/OSDictionary.h>
#include <IOKit/IOLib.h>
#include <IOKit/IOService.h>
#include <IOKit/storage/IOCDMedia.h>
#include <IOKit/IORegistryEntry.h>


//-----------------------------------------------------------------------------
//	Static Function Prototypes
//-----------------------------------------------------------------------------


static QTOCDataFormat10Ptr	CreateBufferFromData 		( OSData * theData );
static IOCDMedia *			GetCDMediaObjectFromName 	( const char * ioBSDNamePtr );


//-----------------------------------------------------------------------------
//	CreateBufferFromIORegistry - Allocates memory for a C-string and copies
//								the contents of the IORegistryEntry to it.
//
//	NB:	The calling function should dispose of the memory by calling IOFree
//-----------------------------------------------------------------------------

QTOCDataFormat10Ptr
CreateBufferFromIORegistry ( mount_t mountPtr )
{

	QTOCDataFormat10Ptr		TOCDataPtr			= NULL;
	OSObject *				objectPtr			= NULL;
	OSData *				dataPtr				= NULL;
	IOCDMedia *				cdMediaPtr			= NULL;
	char *					ioBSDNamePtr		= NULL;

	DebugLog ( ( "CreateBufferFromIORegistry: Entering...\n" ) );
	
	DebugAssert ( ( mountPtr != NULL ) );
	
	ioBSDNamePtr = vfs_statfs ( mountPtr )->f_mntfromname;
	DebugAssert ( ( ioBSDNamePtr != NULL ) );
	
	cdMediaPtr = GetCDMediaObjectFromName ( ioBSDNamePtr );
	DebugAssert ( ( cdMediaPtr != NULL ) );
	
	if ( cdMediaPtr != NULL )
	{
		
		// Get the TOC property
		objectPtr = cdMediaPtr->getProperty ( kIOCDMediaTOCKey );
		if ( objectPtr == NULL )
		{
		
			DebugLog ( ( "CreateBufferFromIORegistry: objectPtr is NULL.\n" ) );
			return NULL;
		
		}
		
		// Cast it to an OSData *
		dataPtr = OSDynamicCast ( OSData, objectPtr );
		if ( dataPtr == NULL )
		{
		
			DebugLog ( ( "CreateBufferFromIORegistry: dataPtr is NULL.\n" ) );
			return NULL;
		
		}

		// Get the data from the registry entry
		TOCDataPtr = CreateBufferFromData ( dataPtr );
		
		DebugLog ( ( "Releasing refcsount on IOCDMedia.\n" ) );
		cdMediaPtr->release ( );
		
	}
	
	DebugLog ( ( "CreateBufferFromIORegistry: exiting...\n" ) );
	
	return TOCDataPtr;
	
}


//-----------------------------------------------------------------------------
//	DisposeBufferFromIORegistry - Frees memory occupied by data structure
//-----------------------------------------------------------------------------

void
DisposeBufferFromIORegistry ( QTOCDataFormat10Ptr TOCDataPtr )
{

	DebugLog ( ( "DisposeBufferFromIORegistry: Entering...\n" ) );

	DebugAssert ( ( TOCDataPtr != NULL ) );
	
	// Free the correct number of bytes. The TOCData has a length word
	// for its first field, so we free the number of bytes specified by
	// the length word, plus the length word itself.
	IOFreeData ( TOCDataPtr,
			( OSSwapBigToHostInt16 ( TOCDataPtr->TOCDataLength ) + sizeof ( TOCDataPtr->TOCDataLength ) ) );
	
	DebugLog ( ( "DisposeBufferFromIORegistry: Exiting...\n" ) );
	
}

#if 0
#pragma mark -
#endif


//-----------------------------------------------------------------------------
//	CreateBufferFromData - 	Allocates memory for a C-string and copies
//							the contents of the IORegistryEntry to it.
//
//	NB:	The calling function should dispose of the memory by calling IOFree
//-----------------------------------------------------------------------------

QTOCDataFormat10Ptr
CreateBufferFromData ( OSData * theData )
{

	vm_size_t          		bufferLength	= 0;
	QTOCDataFormat10Ptr		buffer			= NULL;
	
	DebugLog ( ( "CreateBufferFromData: Entering...\n" ) );
	
	DebugAssert ( ( theData != NULL ) );
	
	if ( theData == NULL )
	{
		
		DebugLog ( ( "CreateBufferFromData: theData is NULL.\n" ) );
		return NULL;
		
	}
	
	bufferLength = theData->getLength ( );
	buffer		 = ( QTOCDataFormat10Ptr ) IOMallocData ( bufferLength );
		
	if ( buffer != NULL )
	{
		
		// Copy the bytes into the buffer
		bcopy ( theData->getBytesNoCopy ( ), buffer, bufferLength );
		
	}
	
	DebugLog ( ( "CreateBufferFromData: exiting.\n" ) );
	
	return buffer;
	
}


//-----------------------------------------------------------------------------
//	GetCDMediaObjectFromName - 	Uses the BSD name to get a reference to the
//								corresponding IOCDMedia object
//-----------------------------------------------------------------------------

IOCDMedia *
GetCDMediaObjectFromName ( const char * ioBSDNamePtr )
{
	
	OSIterator *		iteratorPtr			= NULL;
	IORegistryEntry *	registryEntryPtr	= NULL;
	IOCDMedia *			objectPtr			= NULL;
	OSDictionary *		matchingDictPtr		= NULL;
	
	DebugLog ( ( "GetCDMediaObjectFromName: Entering...\n" ) );
	
	DebugAssert ( ( ioBSDNamePtr != NULL ) );
	
	DebugLog ( ( "GetCDMediaObjectFromName: On enter ioBSDNamePtr = %s.\n", ioBSDNamePtr ) );
	
	// Check to see if we need to strip off any leading stuff
	if ( !strncmp ( ioBSDNamePtr, "/dev/r", 6 ) )
	{

		// Strip off the /dev/r from /dev/rdiskX
		ioBSDNamePtr = &ioBSDNamePtr[6];	

	}

	else if ( !strncmp ( ioBSDNamePtr, "/dev/", 5 ) )
	{

		// Strip off the /dev/ from /dev/diskX
		ioBSDNamePtr = &ioBSDNamePtr[5];	

	}
	
	if ( strncmp ( ioBSDNamePtr, "disk", 4 ) )
	{
		
		// Not in correct format, return NULL
		DebugLog ( ( "GetCDMediaObjectFromName: not in correct format, ioBSDNamePtr = %s.\n", ioBSDNamePtr ) );
		
		return NULL;
		
	}
	
	DebugLog ( ( "GetCDMediaObjectFromName: ioBSDNamePtr = %s.\n", ioBSDNamePtr ) );
	
	// Get a dictionary which describes the bsd device
	matchingDictPtr = IOBSDNameMatching ( ioBSDNamePtr );
	
	// Get an iterator of registry entries
	iteratorPtr = IOService::getMatchingServices ( matchingDictPtr );
	if ( iteratorPtr == NULL )
	{
		
		DebugLog ( ( "GetCDMediaObjectFromName: iteratorPtr is NULL.\n" ) );
		return NULL;
		
	}
	
	// Release the dictionary
	matchingDictPtr->release ( );
			
	DebugLog ( ( "Acquired refcount on iterator and media.\n" ) );
	
	// Get the object out of the iterator (NB: we're guaranteed only one object in the iterator
	// because there is a 1:1 correspondence between BSD Names for devices and IOKit objects
	registryEntryPtr = ( IORegistryEntry * ) iteratorPtr->getNextObject ( );
	if ( registryEntryPtr == NULL )
	{
		
		DebugLog ( ( "GetCDMediaObjectFromName: registryEntryPtr is NULL.\n" ) );
		return NULL;
		
	}
	
	// Cast it to the correct type
	objectPtr = OSDynamicCast ( IOCDMedia, registryEntryPtr );
	if ( objectPtr == NULL )
	{
		
		// Cast failed...spew an error
		DebugLog ( ( "GetCDMediaObjectFromName: objectPtr is NULL, Dynamic Cast failed.\n" ) );
		
	}
	
	DebugLog ( ( "GetCDMediaObjectFromName: exiting...\n" ) );
	
	// Bump the refcount on the CDMedia so that when we release the iterator
	// we still have a refcount on it.
	if ( objectPtr != NULL )
	{
		
		objectPtr->retain ( );
		
	}
	
	// Release the iterator
	iteratorPtr->release ( );	
	
	return ( objectPtr );
	
}


//-----------------------------------------------------------------------------
//				End				Of			File
//-----------------------------------------------------------------------------
