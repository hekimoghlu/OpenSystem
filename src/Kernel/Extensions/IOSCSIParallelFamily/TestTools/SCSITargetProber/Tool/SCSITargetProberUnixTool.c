/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 12, 2022.
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

#include <stdio.h>
#include <stdbool.h>
#include <unistd.h>
#include <mach/mach_error.h>
#include <CoreFoundation/CoreFoundation.h>
#include "Probing.h"


//-----------------------------------------------------------------------------
//	Macros
//-----------------------------------------------------------------------------

#define DEBUG 0

#define DEBUG_ASSERT_COMPONENT_NAME_STRING "SCSITargetProberUnixTool"

#if DEBUG
#define DEBUG_ASSERT_MESSAGE(componentNameString,	\
							 assertionString,		\
							 exceptionLabelString,	\
							 errorString,			\
							 fileName,				\
							 lineNumber,			\
							 errorCode)				\
DebugAssert(componentNameString,					\
					   assertionString,				\
					   exceptionLabelString,		\
					   errorString,					\
					   fileName,					\
					   lineNumber,					\
					   errorCode)					\

static void
DebugAssert ( const char *	componentNameString,
			  const char *	assertionString,
			  const char *	exceptionLabelString,
			  const char *	errorString,
			  const char *	fileName,
			  long			lineNumber,
			  int			errorCode )
{
	
	if ( ( assertionString != NULL ) && ( *assertionString != '\0' ) )
		printf ( "Assertion failed: %s: %s\n", componentNameString, assertionString );
	else
		printf ( "Check failed: %s:\n", componentNameString );
	if ( exceptionLabelString != NULL )
		printf ( "	 %s\n", exceptionLabelString );
	if ( errorString != NULL )
		printf ( "	 %s\n", errorString );
	if ( fileName != NULL )
		printf ( "	 file: %s\n", fileName );
	if ( lineNumber != 0 )
		printf ( "	 line: %ld\n", lineNumber );
	if ( errorCode != 0 )
		printf ( "	 error: %d\n", errorCode );
	
}

#endif	/* DEBUG */

#include <AssertMacros.h>


//-----------------------------------------------------------------------------
//	Prototypes
//-----------------------------------------------------------------------------

static IOReturn
ParseArguments ( int argc, const char * argv[],
				 UInt64 * domainID, SCSITargetIdentifier * targetID );

static void
PrintUsage ( void );


//-----------------------------------------------------------------------------
//	main - Our main entry point
//-----------------------------------------------------------------------------

int
main ( int argc, const char * argv[] )
{
	
	int						returnCode	= 0;
	IOReturn				result		= kIOReturnSuccess;
	UInt64					domainID	= 0;
	SCSITargetIdentifier	targetID	= 0;
	
	result = ParseArguments ( argc, argv, &domainID, &targetID );
	require_action ( ( result == 0 ), ErrorExit, PrintUsage ( ); returnCode = 1 );
	
	printf ( "SCSITargetProber: Probing device for domain = %lld, targetID = %lld\n", domainID, targetID );
	
	result = ReprobeDomainTarget ( domainID, targetID );
	require_action ( ( result == 0 ), ErrorExit, printf ( "Error = %s (0x%08x) reprobing device\n", mach_error_string ( result ), result ); returnCode = 2 );
	
	return 0;
	
	
ErrorExit:
	
	
	return returnCode;
	
}


//-----------------------------------------------------------------------------
//	ParseArguments - Parses argument list
//-----------------------------------------------------------------------------

static IOReturn
ParseArguments ( int argc, const char * argv[],
				 UInt64 * domainID, SCSITargetIdentifier * targetID )
{
	
	IOReturn	result	= kIOReturnSuccess;
	int			ch;
	
	while ( ( ch = getopt ( argc, ( char * const * ) argv, "d:t:" ) ) != -1 )
	{
		
		switch ( ch )
		{
			
			case 'd':
				*domainID = strtoull ( optarg, ( char ** ) NULL, 10 );
				break;
			
			case 't':
				*targetID = strtoull ( optarg, ( char ** ) NULL, 10 );
				break;
			
			default:
				result = kIOReturnBadArgument;
				break;
			
		}
		
	}
	
	return result;
	
}


//-----------------------------------------------------------------------------
//	PrintUsage - Prints out usage
//-----------------------------------------------------------------------------

void
PrintUsage ( void )
{
	
	printf ( "\n" );
	printf ( "Usage: stp -d domainID -t targetID\n" );
	printf ( "\t\t" );
	printf ( "-d This option specifies which SCSI Domain on which to find the target for probing\n" );
	printf ( "\t\t" );
	printf ( "-t This option specifices which SCSI Target Identifier should be probed\n" );
	printf ( "\n" );
	
}