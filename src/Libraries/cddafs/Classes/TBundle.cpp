/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 3, 2023.
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

#include "TBundle.h"
#include "TSystemUtils.h"


//-----------------------------------------------------------------------------
//	Macros
//-----------------------------------------------------------------------------

#define DEBUG	0

#ifndef DEBUG_ASSERT_COMPONENT_NAME_STRING
	#define DEBUG_ASSERT_COMPONENT_NAME_STRING "TBundle"
#endif

#include <AssertMacros.h>

#define kLocalizableString		"Localizable"
#define kStringsTypeString		"strings"
#define kEmptyString			""


//-----------------------------------------------------------------------------
//	Constructor														 [PUBLIC]
//-----------------------------------------------------------------------------

TBundle::TBundle ( CFBundleRef bundle ) :
	fLocalizationDictionaryForTable ( NULL )
{
	
	check ( bundle );
	fCFBundleRef = ( CFBundleRef ) ::CFRetain ( bundle );
	
}


//-----------------------------------------------------------------------------
//	Destructor														 [PUBLIC]
//-----------------------------------------------------------------------------

TBundle::~TBundle ( void )
{
	
	check ( fCFBundleRef );
	::CFRelease ( fCFBundleRef );
	fCFBundleRef = NULL;
	
	if ( fLocalizationDictionaryForTable != NULL )
	{
		
		::CFRelease ( fLocalizationDictionaryForTable );
		fLocalizationDictionaryForTable = NULL;
		
	}
	
}


//-----------------------------------------------------------------------------
//	CopyLocalizedStringForKey										 [PUBLIC]
//-----------------------------------------------------------------------------

CFStringRef
TBundle::CopyLocalizedStringForKey ( CFStringRef key,
									 CFStringRef defaultValue,
									 CFStringRef tableName )
{
	
	CFStringRef		result = NULL;

	if ( tableName == NULL )
	{
		tableName = CFSTR ( kLocalizableString );
	}
	
	if ( fLocalizationDictionaryForTable == NULL )
	{
		fLocalizationDictionaryForTable = CopyLocalizationDictionaryForTable ( tableName );
	}
	
	if ( fLocalizationDictionaryForTable != NULL )
	{
		result = ( CFStringRef ) ::CFDictionaryGetValue ( fLocalizationDictionaryForTable, key );
		::CFRetain ( result );
	}
	
	if ( ( result == NULL ) && ( defaultValue != NULL ) )
	{
		result = ( CFStringRef ) ::CFRetain ( defaultValue );
	}
	
	return result;
	
}


//-----------------------------------------------------------------------------
//	CopyLocalizationDictionaryForTable								[PRIVATE]
//-----------------------------------------------------------------------------

CFDictionaryRef
TBundle::CopyLocalizationDictionaryForTable ( CFStringRef table )
{
	
	CFDictionaryRef		stringTable				= NULL;
	CFURLRef			localizedStringsURL		= NULL;
	CFDataRef           tableData				= NULL;
	CFStringRef			errStr					= NULL;
    
	localizedStringsURL = CopyURLForResourceOfTypeInBundle (
				table,
				CFSTR ( kStringsTypeString ),
				fCFBundleRef );

	require ( ( localizedStringsURL != NULL ), ErrorExit );

    tableData = TSystemUtils::ReadDataFromURL ( localizedStringsURL );
    
	require ( ( tableData != NULL ), ReleaseURL );
	
	stringTable = ( CFDictionaryRef ) ::CFPropertyListCreateFromXMLData (
											kCFAllocatorDefault,
											tableData,
											kCFPropertyListImmutable,
											&errStr );

	if ( errStr != NULL )
	{
	
	#if DEBUG
		::CFShow ( errStr );
	#endif
		
		::CFRelease ( errStr );
		errStr = NULL;
		
	}
	
	::CFRelease ( tableData );
	tableData = NULL;
	
	check ( stringTable != NULL );
	
    
ReleaseURL:
	
	
	::CFRelease ( localizedStringsURL );
	localizedStringsURL = NULL;
	
	
ErrorExit:
	
	
    return stringTable;
	
}


//-----------------------------------------------------------------------------
//	CopyLocalizations												[PRIVATE]
//-----------------------------------------------------------------------------

CFArrayRef
TBundle::CopyLocalizations ( void )
{
	return ::CFBundleCopyBundleLocalizations ( fCFBundleRef );
}


//-----------------------------------------------------------------------------
//	CopyLocalizationsForPrefs										[PRIVATE]
//-----------------------------------------------------------------------------

CFArrayRef
TBundle::CopyLocalizationsForPrefs ( CFArrayRef bundleLocalizations,
									 CFArrayRef preferredLanguages )
{
	return ::CFBundleCopyLocalizationsForPreferences ( bundleLocalizations, preferredLanguages );
}


//-----------------------------------------------------------------------------
//	CopyURLForResourceOfTypeInBundle								[PRIVATE]
//-----------------------------------------------------------------------------

CFURLRef
TBundle::CopyURLForResourceOfTypeInBundle ( CFStringRef		resource,
										    CFStringRef		type,
										    CFBundleRef 	bundle )
{
	
	CFURLRef			result					= NULL;
	CFArrayRef			preferredLanguages		= NULL;
	CFArrayRef			bundleLocalizations		= NULL;
	CFArrayRef			preferredLocalizations	= NULL;
	CFIndex				index					= 0;
	CFIndex				count					= 0;

	if ( bundle == NULL )
	{
		bundle = ::CFBundleGetMainBundle ( );
	}
	
	require ( ( bundle != NULL ), ErrorExit );
	
	preferredLanguages = TSystemUtils::GetPreferredLanguages ( );
	require ( ( preferredLanguages != NULL ), ErrorExit );
	
	bundleLocalizations		= CopyLocalizations ( );
	preferredLocalizations	= CopyLocalizationsForPrefs ( bundleLocalizations, preferredLanguages );

	count = ::CFArrayGetCount ( preferredLocalizations );

	for ( index = 0; ( result == NULL ) && ( index < count ); index++)
	{
		
		CFStringRef	item = ( CFStringRef ) ::CFArrayGetValueAtIndex ( preferredLocalizations, index );
		
		result = CopyURLForResource ( resource,
									  type,
									  NULL,
									  item );
		
	}
	
	if ( result == NULL )
	{
		
		CFStringRef	developmentLocalization = ::CFBundleGetDevelopmentRegion ( fCFBundleRef );
		
		if ( developmentLocalization != NULL )
		{
			result = CopyURLForResource ( resource, type, NULL, developmentLocalization );
		}
		
	}

	if ( preferredLocalizations != NULL )
		::CFRelease ( preferredLocalizations );
    
	if ( preferredLanguages != NULL )
		::CFRelease ( preferredLanguages );
	
	if ( bundleLocalizations != NULL )
		::CFRelease ( bundleLocalizations );
	
	
ErrorExit:
	
	
	return result;
	
}


//-----------------------------------------------------------------------------
//	CopyURLForResource												[PRIVATE]
//-----------------------------------------------------------------------------

CFURLRef
TBundle::CopyURLForResource ( CFStringRef resource,
							  CFStringRef type,
							  CFStringRef dir,
							  CFStringRef localization )
{
	
	CFURLRef		resultURL 	= NULL;
	
	resultURL = ::CFBundleCopyResourceURLForLocalization ( fCFBundleRef,
														   resource,
														   type,
														   dir,
														   localization );
	
	return resultURL;
	
}


//-----------------------------------------------------------------------------
//					End				Of			File
//-----------------------------------------------------------------------------
