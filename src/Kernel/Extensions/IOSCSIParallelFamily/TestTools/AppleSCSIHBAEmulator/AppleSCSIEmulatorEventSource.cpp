/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 3, 2023.
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

#include "DebugSupport.h"
#include "AppleSCSIEmulatorEventSource.h"
#include "AppleSCSIEmulatorAdapter.h"


// Define superclass
#define super IOEventSource
OSDefineMetaClassAndStructors ( AppleSCSIEmulatorEventSource, IOEventSource )


//-----------------------------------------------------------------------------
//	Create
//-----------------------------------------------------------------------------

AppleSCSIEmulatorEventSource *
AppleSCSIEmulatorEventSource::Create ( OSObject * owner, Action action )
{
	
	AppleSCSIEmulatorEventSource *	es		= NULL;
	bool							result	= false;
	
	es = OSTypeAlloc ( AppleSCSIEmulatorEventSource );
	require_nonzero ( es, ErrorExit );
	
	result = es->Init ( owner, action );
	require ( result, ReleaseEventSource );
	
	return es;
	
	
ReleaseEventSource:
	
	
	es->release ( );
	
	
ErrorExit:
	
	
	return NULL;
	
}


//-----------------------------------------------------------------------------
//	Init
//-----------------------------------------------------------------------------

bool
AppleSCSIEmulatorEventSource::Init ( OSObject * owner, Action action )
{
	
	bool	result = false;
	
	// Initialize the queue head.
	queue_init ( &fResponderQueue );
	
	fLock = IOSimpleLockAlloc ( );
	if ( fLock != NULL )
	{
		
		// Call the superclass.
		result = super::init ( owner, ( IOEventSource::Action ) action );
		
	}
	
	return result;
	
}


//-----------------------------------------------------------------------------
//	AddItemToQueue
//-----------------------------------------------------------------------------

void
AppleSCSIEmulatorEventSource::AddItemToQueue ( SCSIEmulatorRequestBlock * srb )
{
	
	// Take the lock to protect the queue.
	IOSimpleLockLock ( fLock );
	
	// Add the item to the queue.
	queue_enter ( &fResponderQueue, srb, SCSIEmulatorRequestBlock *, fQueueChain );
	
	// Drop the lock.
	IOSimpleLockUnlock ( fLock );
	
	// Wakeup the thread since there's work to do...
	signalWorkAvailable ( );
	
}


//-----------------------------------------------------------------------------
//	RemoveItemFromQueue
//-----------------------------------------------------------------------------

SCSIEmulatorRequestBlock *
AppleSCSIEmulatorEventSource::RemoveItemFromQueue ( void )
{
	
	SCSIEmulatorRequestBlock *	srb = NULL;
	
	// Take the lock to protect the queue.
	IOSimpleLockLock ( fLock );
	
	if ( !queue_empty ( &fResponderQueue ) )
	{
		
		// Remove the item to the queue.
		queue_remove_first ( &fResponderQueue, srb, SCSIEmulatorRequestBlock *, fQueueChain );
	
	}
	
	// Drop the lock.
	IOSimpleLockUnlock ( fLock );
	
	return srb;
	
}


//-----------------------------------------------------------------------------
//	checkForWork
//-----------------------------------------------------------------------------

bool
AppleSCSIEmulatorEventSource::checkForWork ( void )
{
	
	SCSIEmulatorRequestBlock *	srb = NULL;
	
	srb = RemoveItemFromQueue ( );
	while ( srb != NULL )
	{
		
		// Complete it if it's non-NULL.
		( *action ) ( owner, srb->fParallelRequest );
		
		srb = RemoveItemFromQueue ( );
		
	}
	
	return false;
	
}
