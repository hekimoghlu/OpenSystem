/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 17, 2023.
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
//-----------------------------------------------------------------------------
//	Includes
//-----------------------------------------------------------------------------

#define DEBUG_ASSERT_COMPONENT_NAME_STRING 					"TSystemUtils"
#include <AssertMacros.h>

#include "TSystemUtils.h"

#include <unistd.h>
#include <sys/stat.h>
#include <SystemConfiguration/SystemConfiguration.h>


//-----------------------------------------------------------------------------
//	Macros
//-----------------------------------------------------------------------------

#define DEBUG	0

#define kAppleLanguagesString	"AppleLanguages"
#define kEmptyString			""


//-----------------------------------------------------------------------------
// GetPreferredLanguages - Gets preferred languages of console user.
//															 [STATIC][PUBLIC]
//-----------------------------------------------------------------------------

CFArrayRef
TSystemUtils::GetPreferredLanguages ( void )
{
	
	CFArrayRef			preferredLanguages	= NULL;
	CFPropertyListRef	languages			= NULL;
	CFStringRef			userName			= NULL;
	CFComparisonResult	equal				= kCFCompareEqualTo;
	uid_t				uid					= 0;

	uid = FindUIDToUse ( );
	seteuid ( uid );
	
	userName = ::CFCopyUserName ( );
	require ( ( userName != NULL ), ErrorExit );
	
	equal = ::CFStringCompare ( userName, CFSTR ( kEmptyString ), 0 );
	require ( ( equal != kCFCompareEqualTo ), ReleaseExit );
	
	languages = ::CFPreferencesCopyValue ( CFSTR ( kAppleLanguagesString ),
										   kCFPreferencesAnyApplication,
										   userName,
										   kCFPreferencesAnyHost );
	
	require ( ( languages != NULL ), ReleaseExit );
	require_action ( ( ::CFGetTypeID ( languages ) == ::CFArrayGetTypeID ( ) ),
					 ReleaseExit,
					 ::CFRelease ( languages ) );
	
	preferredLanguages = ( CFArrayRef ) languages;


ReleaseExit:

	
    ::CFRelease ( userName );


ErrorExit:
	
    
	seteuid ( 0 );
	
	return preferredLanguages;
	
}


//-----------------------------------------------------------------------------
// FindUIDToUse - Finds the UID of the console user.		 [STATIC][PUBLIC]
//-----------------------------------------------------------------------------

uid_t
TSystemUtils::FindUIDToUse ( void )
{
	
	uid_t 				uid			= 0;
	gid_t				gid			= 0;
	CFStringRef			userName	= NULL;
	SCDynamicStoreRef	storeRef	= NULL;

	storeRef = ::SCDynamicStoreCreate ( kCFAllocatorDefault,
										CFSTR ( "cddafs.util" ),
										NULL,
										NULL );
	require ( ( storeRef != NULL ), ErrorExit );

	userName = ::SCDynamicStoreCopyConsoleUser ( storeRef,
												 &uid,
												 &gid );
	require ( ( userName != NULL ), ReleaseDynamicStore );
	::CFRelease ( userName );
	
	
ReleaseDynamicStore:
	
	
	require_quiet ( ( storeRef != NULL ), ErrorExit );
	::CFRelease ( storeRef );
	storeRef = NULL;
	
	
ErrorExit:
	
	
	return uid;
	
}


//-----------------------------------------------------------------------------
// ReadDataFromURL - Reads data given a file URL.		 [STATIC][PUBLIC]
//-----------------------------------------------------------------------------

CFDataRef
TSystemUtils::ReadDataFromURL ( CFURLRef url )
{

    CFMutableDataRef    data            = NULL;
    Boolean				result			= false;
	CFNumberRef         fileSizeNumber  = NULL;
    CFIndex             fileSize        = 0;
    UInt8 *             dataPtr         = NULL;
    UInt8 *             endPtr          = NULL;
    CFReadStreamRef     readStream      = NULL;
    CFIndex             bytesRead       = 0;
    
    result = CFURLCopyResourcePropertyForKey ( url, kCFURLFileSizeKey, &fileSizeNumber, NULL );
    require ( result, ErrorExit );
    require ( fileSizeNumber != NULL, ErrorExit );
    
    result = CFNumberGetValue ( fileSizeNumber, kCFNumberCFIndexType, &fileSize );
    require ( result, ReleaseNumber );
    
    data = CFDataCreateMutable ( kCFAllocatorDefault, fileSize );
    require ( data, ReleaseNumber );
    
    CFDataSetLength ( data, fileSize );
    
    dataPtr = CFDataGetMutableBytePtr ( data );
    require ( dataPtr, ReleaseNumber );
    
    readStream = CFReadStreamCreateWithFile ( kCFAllocatorDefault, url );
    require ( readStream, ErrorExit );
    
    result = CFReadStreamOpen ( readStream );
    require ( result, ReleaseStream );
    
    endPtr  = ( UInt8 * ) dataPtr + fileSize;
    
    while ( dataPtr < endPtr )
    {

        bytesRead = CFReadStreamRead ( readStream, dataPtr, endPtr - dataPtr );

        if ( bytesRead > 0 )
        {
            
            dataPtr += bytesRead;
            
        }
        
    }
    
    CFReadStreamClose ( readStream );
    
    
ReleaseStream:
    
    
    CFRelease ( readStream );
    readStream = NULL;
    
    
ReleaseNumber:
    
    
    CFRelease ( fileSizeNumber );
    fileSizeNumber = NULL;
    
    
ErrorExit:
    
    
    return data;
    
}

//-----------------------------------------------------------------------------
//					End				Of			File
//-----------------------------------------------------------------------------
