/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 20, 2025.
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

#include <IOKit/IOTypes.h>
#include "SCSIParallelWorkLoop.h"


//-----------------------------------------------------------------------------
//	Macros
//-----------------------------------------------------------------------------

#define DEBUG 												0
#define DEBUG_ASSERT_COMPONENT_NAME_STRING					"SPI WorkLoop"

#if DEBUG
#define SCSI_PARALLEL_WORKLOOP_DEBUGGING_LEVEL				0
#endif


#include "IOSCSIParallelFamilyDebugging.h"


#if ( SCSI_PARALLEL_WORKLOOP_DEBUGGING_LEVEL >= 1 )
#define PANIC_NOW(x)		panic x
#else
#define PANIC_NOW(x)
#endif

#if ( SCSI_PARALLEL_WORKLOOP_DEBUGGING_LEVEL >= 2 )
#define ERROR_LOG(x)		IOLog x
#else
#define ERROR_LOG(x)
#endif

#if ( SCSI_PARALLEL_WORKLOOP_DEBUGGING_LEVEL >= 3 )
#define STATUS_LOG(x)		IOLog x
#else
#define STATUS_LOG(x)
#endif


#define super IOWorkLoop
OSDefineMetaClassAndStructors ( SCSIParallelWorkLoop, IOWorkLoop );


#if 0
#pragma mark -
#pragma mark IOKit Member Routines
#pragma mark -
#endif


//-----------------------------------------------------------------------------
//	Create													   [STATIC][PUBLIC]
//-----------------------------------------------------------------------------

SCSIParallelWorkLoop *
SCSIParallelWorkLoop::Create ( const char *	lockGroupName )
{
	
	SCSIParallelWorkLoop *		workLoop = NULL;
	
	workLoop = OSTypeAlloc ( SCSIParallelWorkLoop );
	require_nonzero ( workLoop, ErrorExit );
	
	require ( workLoop->InitWithLockGroupName ( lockGroupName ), ReleaseWorkLoop );
	
	return workLoop;
	
	
ReleaseWorkLoop:
	
	
	require_nonzero ( workLoop, ErrorExit );
	workLoop->release ( );
	workLoop = NULL;
	
	
ErrorExit:
	
	
	return workLoop;
	
}


//-----------------------------------------------------------------------------
//	InitWithLockGroupName											[PROTECTED]
//-----------------------------------------------------------------------------

bool
SCSIParallelWorkLoop::InitWithLockGroupName ( const char * lockGroupName )
{
	
	bool	result = false;
	
	fLockGroup = lck_grp_alloc_init ( lockGroupName, LCK_GRP_ATTR_NULL );
	require_nonzero ( fLockGroup, ErrorExit );
	
	// Allocate the gateLock before calling the super class. This allows
	// us to profile contention on our workloop lock.
	gateLock = IORecursiveLockAllocWithLockGroup ( fLockGroup );
	
	result = super::init ( );
	
	
ErrorExit:
	
	
	return result;
	
}


//-----------------------------------------------------------------------------
//	free															[PROTECTED]
//-----------------------------------------------------------------------------

void
SCSIParallelWorkLoop::free ( void )
{
	
	// NOTE: IOWorkLoop::free() gets called multiple times!
	if ( fLockGroup != NULL )
	{
		
		lck_grp_free ( fLockGroup );
		fLockGroup = NULL;
		
	}
	
	super::free ( );
	
}
