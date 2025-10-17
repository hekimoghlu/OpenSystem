/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 8, 2023.
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
#include "IOFWUserPHYPacketListener.h"

#import <IOKit/firewire/IOFireWireController.h>
#import <IOKit/firewire/IOFireWireNub.h>
#import <IOKit/firewire/IOFWPHYPacketListener.h>

// ============================================================
//	IOFWUserPHYPacketListener methods
// ============================================================

OSDefineMetaClassAndStructors( IOFWUserPHYPacketListener, IOFWPHYPacketListener );

// withUserClient
//
//

IOFWUserPHYPacketListener *
IOFWUserPHYPacketListener::withUserClient( IOFireWireUserClient * inUserClient, UInt32 queue_count )
{
	IOFWUserPHYPacketListener*	result	= NULL;
	
	result = OSTypeAlloc( IOFWUserPHYPacketListener );
	if( result && !result->initWithUserClient( inUserClient, queue_count ) )
	{
		result->release();
		result = NULL;
	}

	return result;
}

// init
//
//

bool 
IOFWUserPHYPacketListener::initWithUserClient( IOFireWireUserClient * inUserClient, UInt32 queue_count )
{
	bool success = true;
	
	IOFireWireController * control = inUserClient->getOwner()->getController();
	if( !IOFWPHYPacketListener::initWithController( control ) )
		success = false;
	
	if( success )
	{
		fUserClient = inUserClient;
	
		fMaxQueueCount = queue_count;
		
		// enforce a minimum queue size
		if( fMaxQueueCount < 2 )
		{
			fMaxQueueCount = 2;
		}
		
		fAllocatedQueueCount = 0;
		
		fElementWaitingCompletion = NULL;

		// get a lock for the packet queue
		fLock = IOLockAlloc();
		if( fLock == NULL )
		{
			success = false;
		}
	}
	
	return success;
}

// free
//
//

void 
IOFWUserPHYPacketListener::free( void )
{	
	destroyAllElements();
	
	if( fLock )
	{
		IOLockFree( fLock );
		fLock = NULL;
	}
	
	IOFWPHYPacketListener::free();
}

// exporterCleanup
//
//

void
IOFWUserPHYPacketListener::exporterCleanup( const OSObject * self )
{
	IOFWUserPHYPacketListener * me = (IOFWUserPHYPacketListener*)self;
	
	DebugLog("IOFWUserPHYPacketListener::exporterCleanup\n");
	
	me->deactivate();
}

#pragma mark -
/////////////////////////////////////////////////////////////////////////////////

// setPacketCallback
//
//

IOReturn
IOFWUserPHYPacketListener::setPacketCallback(	OSAsyncReference64		async_ref,
												mach_vm_address_t		callback,
												io_user_reference_t		refCon )
{
//	kprintf( "IOFWUserPHYPacketListener::setPacketCallback - asyncref = 0x%016llx callback = 0x%016llx, refcon = 0x%016llx\n", async_ref[0], callback, refCon );
	
	if( callback )
	{
		IOFireWireUserClient::setAsyncReference64( fCallbackAsyncRef, (mach_port_t)async_ref[0], callback, refCon );
	}
	else
	{
		fCallbackAsyncRef[0] = 0;
	}
	
	return kIOReturnSuccess;
}

// setSkippedCallback
//
//

IOReturn
IOFWUserPHYPacketListener::setSkippedCallback(	OSAsyncReference64		async_ref,
												mach_vm_address_t		callback,
												io_user_reference_t		refCon )
{
//	kprintf( "IOFWUserPHYPacketListener::setSkippedCallback - asyncref = 0x%016llx callback = 0x%016llx, refcon = 0x%016llx\n", async_ref[0], callback, refCon );
	
	if( callback )
	{
		IOFireWireUserClient::setAsyncReference64( fSkippedAsyncRef, (mach_port_t)async_ref[0], callback, refCon );
	}
	else
	{
		fSkippedAsyncRef[0] = 0;
	}
	
	return kIOReturnSuccess;
}

// processPHYPacket
//
// on workloop

void IOFWUserPHYPacketListener::processPHYPacket( UInt32 data1, UInt32 data2 )
{
	IOLockLock( fLock );

	PHYRxElement * element = NULL;
	
//	kprintf( "IOFWUserPHYPacketListener::processPHYPacket - 0x%08lx %08lx\n", data1, data2 );
	
	// try to get a data element
	
	// allocate an element, but tell the allocator to reserve an element for skipped packets
	element = allocateDataElement();
	if( element )
	{
		// we've got a data element
		element->type = kTypeData;
		element->data1 = data1;
		element->data2 = data2;

		// tell user space if it's not already busy
		if( fElementWaitingCompletion == NULL )
		{
		//	kprintf( "IOFWUserPHYPacketListener::processPHYPacket - data element - send notification\n" );

			sendPacketNotification( element );
		}
		else
		{
		//	kprintf( "IOFWUserPHYPacketListener::processPHYPacket - data element - queue - fElementWaitingCompletion = 0x%08lx\n", fElementWaitingCompletion );

			// else queue it up
			addElementToPendingList( element );
		}
	}
	else
	{
		// if we couldn't get a data element we're skipping

		// if we're skipping, but user space isn't working on anything
		// then we're in trouble
		FWKLOGASSERT( fElementWaitingCompletion != NULL );

		// reuse the tail if it's a skip packet
		if( fPendingListTail && fPendingListTail->type == kTypeSkipped )
		{
			element = fPendingListTail;

			// and it hasn't been sent to the user yet
			// this should be true because the queue size is always at least 2 and the 
			// commands must be processed by user space sequentially
			FWKLOGASSERT( fPendingListTail != fElementWaitingCompletion );

			// bump the count
			UInt32 count = element->getSkippedCount();
			count++;
			element->setSkippedCount( count );
		}
		else
		{
			// else get an element
			element = allocateElement();
			if( element )
			{
				element->type = kTypeSkipped;
				
				// set our skipped count
				element->setSkippedCount( 1 );

		//		kprintf( "IOFWUserPHYPacketListener::processPHYPacket - skip element - queue\n" );

				// queue it up
				addElementToPendingList( element );
			}
			else
			{
				// just drop it I guess
				IOLog( "FireWire - UserPHYPacketListener out of elements\n" );
			}
		}
	}

	IOLockUnlock( fLock );	
}

// clientCommandIsComplete
//
// on user thread

void 
IOFWUserPHYPacketListener::clientCommandIsComplete( FWClientCommandID commandID )
{
	IOLockLock( fLock );

//	kprintf( "IOFWUserPHYPacketListener::clientCommandIsComplete - commandID = 0x%08lx\n", commandID );

	// verify we're completing the right command
	if( fElementWaitingCompletion == commandID )
	{
		// we're done with the outstanding element
		deallocateElement( fElementWaitingCompletion );
	
		fElementWaitingCompletion = NULL;

	//	kprintf( "IOFWUserPHYPacketListener::clientCommandIsComplete - element = 0x%08lx fElementWaitingCompletion = 0x%08lx\n", commandID, fElementWaitingCompletion );
			
		// if we've got another one on the pending list, send it off
		PHYRxElement * element = fPendingListHead;
		if( element )
		{
			removeElementFromPendingList( element );
			sendPacketNotification( element );
		}
	}
	
	IOLockUnlock( fLock );
}

// sendPacketNotification
//
// lock is held

void
IOFWUserPHYPacketListener::sendPacketNotification( IOFWUserPHYPacketListener::PHYRxElement * element )
{	
	if( fElementWaitingCompletion == NULL )
	{
		if( element->type == kTypeData )
		{
			fElementWaitingCompletion = element;
			
			io_user_reference_t args[3];
			args[0] = (io_user_reference_t)element;		// commandID
			args[1] = element->data1;					// data1
			args[2] = element->data2;					// data2
	
		//	kprintf( "IOFWUserPHYPacketListener::sendPacketNotification - kTypeData fElementWaitingCompletion = 0x%08lx\n", fElementWaitingCompletion );
	
		//	kprintf( "IOFWUserPHYPacketListener::sendPacketNotification - fCallbackAsyncRef[0] = 0x%016llx\n", fCallbackAsyncRef[0] );

			IOFireWireUserClient::sendAsyncResult64( fCallbackAsyncRef, kIOReturnSuccess, args, 3 );
		}
		else if( element->type == kTypeSkipped )
		{
			fElementWaitingCompletion = element;
			
			io_user_reference_t args[3];
			args[0] = (io_user_reference_t)element;		// commandID
			args[1] = element->getSkippedCount();		// count
			args[2] = 0;								//zzz for some reason I need to send an arg count of 3 for my data to make it to user space 
														//zzz sounds like a kernel bug. pad it for now
														
		//	kprintf( "IOFWUserPHYPacketListener::sendPacketNotification - kTypeSkipped count = %d, fElementWaitingCompletion = 0x%08lx\n", element->getSkippedCount(), (UInt32)fElementWaitingCompletion );

		//	kprintf( "IOFWUserPHYPacketListener::sendPacketNotification - fSkippedAsyncRef[0] = 0x%016llx\n", fSkippedAsyncRef[0] );
		
			IOFireWireUserClient::sendAsyncResult64( fSkippedAsyncRef, kIOReturnSuccess, args, 3 );
		}
	}
}

#pragma mark -

// allocateElement
//
//

IOFWUserPHYPacketListener::PHYRxElement * IOFWUserPHYPacketListener::allocateElement( void )
{
	//
	// allocate
	//
	
	PHYRxElement * element = fFreeListHead;
	
	if( element == NULL )
	{
		// create elements on demand
		// make skipped elements up to the threshold
		if(fAllocatedQueueCount < fMaxQueueCount )
		{
			element = new PHYRxElement;
			if( element != NULL )
			{
				element->next = NULL;
				element->prev = NULL;
				element->type = kTypeNone;
				element->state = kFreeState;
				element->data1 = 0;
				element->data2 = 0;
				
				//
				// link it in
				//
				
				fFreeListHead = element;
				fFreeListTail = element;
			
				fAllocatedQueueCount++;
			}
		}
	}
	
	if( element != NULL )
	{
		fFreeListHead = element->next;
		if( fFreeListHead )
		{
			fFreeListHead->prev = NULL;
		}
		else
		{
			FWKLOGASSERT( fFreeListTail == element );

			fFreeListTail = NULL;
		}
		
		FWKLOGASSERT( element->prev == NULL );
		FWKLOGASSERT( element->state == kFreeState );
		 
		element->next = NULL;
		element->prev = NULL;
		element->state = kFreeState;
		
		DebugLog( "IOFWUserPHYPacketListener::allocateElement - element = %p\n", element );
	}
	
	return element;
}

// allocateElement
//
//

IOFWUserPHYPacketListener::PHYRxElement * IOFWUserPHYPacketListener::allocateDataElement( void )
{
	//
	// allocate
	//
	
	PHYRxElement * element = fFreeListHead;
	
	if( element == NULL )
	{
		// create elements on demand
		// make data elements only if we can make one more for skipped packets

		if( fAllocatedQueueCount < (fMaxQueueCount - 1) )
		{
			element = new PHYRxElement;
			if( element != NULL )
			{
				element->next = NULL;
				element->prev = NULL;
				element->type = kTypeNone;
				element->state = kFreeState;
				element->data1 = 0;
				element->data2 = 0;
				
				//
				// link it in
				//
				
				fFreeListHead = element;
				fFreeListTail = element;
			
				fAllocatedQueueCount++;
			}
		}
	}
	
	if( element != NULL )
	{
		// we cannot allocate the last element if it is a data element
		if( (element != fFreeListTail) ||							// if it's not the tail
			(fAllocatedQueueCount < (fMaxQueueCount - 1))  )		// or we can allocate more
		{
			fFreeListHead = element->next;
			if( fFreeListHead )
			{
				fFreeListHead->prev = NULL;
			}
			else
			{
				FWKLOGASSERT( fFreeListTail == element );

				fFreeListTail = NULL;
			}
			
			FWKLOGASSERT( element->prev == NULL );
			FWKLOGASSERT( element->state == kFreeState );
			 
			element->next = NULL;
			element->prev = NULL;
			element->state = kFreeState;
			
			DebugLog( "IOFWUserPHYPacketListener::allocateDataElement - element = %p\n", element );
		}
		else
		{
			element = NULL;
		}
	}
	
	return element;
}


// deallocateElement
//
//

void IOFWUserPHYPacketListener::deallocateElement( PHYRxElement * element )
{	
	DebugLog( "IOFWUserPHYPacketListener::deallocateElement - element = %p\n", element );
	
	element->next = NULL;
	element->prev = fFreeListTail;
	element->state = kFreeState;
	
	if( fFreeListTail )
	{
		fFreeListTail->next = element;	
	}
	else
	{
		FWKLOGASSERT( fFreeListHead == NULL );
		
		fFreeListHead = element;
	}
	
	fFreeListTail = element;
	
	FWKLOGASSERT( fFreeListHead != NULL );
	FWKLOGASSERT( fFreeListTail != NULL );	
}

#pragma mark -

// destroyAllElements
//
//

void IOFWUserPHYPacketListener::destroyAllElements( void )
{
	DebugLog(( "IOFWUserPHYPacketListener::destroyAllElements\n" ));

	//
	// return all elements to the free pool
	//
	
	{
		PHYRxElement * element = fPendingListHead;
		
		while( element )
		{
			PHYRxElement * next_element = element->next;
					
			removeElementFromPendingList( element );
			
			deallocateElement( element );
			
			element = next_element;
		}
		
		FWKLOGASSERT( fPendingListHead == NULL );
		fPendingListHead = NULL;		// should already be NULL

		FWKLOGASSERT( fPendingListTail == NULL );	
		fPendingListTail = NULL;		// should already be NULL
	}
	
	//
	// delete all elements in free pool
	//
	
	{
		PHYRxElement * element = fFreeListHead;
		while( element )
		{
			PHYRxElement * next_element = element->next;

			delete element;
			
			element = next_element;
		}
		
		fFreeListHead = 0;
		fFreeListTail = 0;
	}
}

#pragma mark -

// addElementToPendingList
//
//

void IOFWUserPHYPacketListener::addElementToPendingList( PHYRxElement * element )
{
	DebugLog( "IOFWUserPHYPacketListener::addElementToPendingList - element = %p\n", element );
	
	// pending should only be entered from the free state
	
	FWKLOGASSERT( element != NULL );
	FWKLOGASSERT( element->next == NULL );
	FWKLOGASSERT( element->state == kFreeState );

	element->next = NULL;
	element->prev = fPendingListTail;
	element->state = kPendingState;
	
	if( fPendingListTail )
	{
		fPendingListTail->next = element;	
	}
	else
	{
		FWKLOGASSERT( fPendingListHead == NULL );

		fPendingListHead = element;
	}
	
	fPendingListTail = element;
	
	FWKLOGASSERT( fPendingListHead != NULL );
	FWKLOGASSERT( fPendingListTail != NULL );
	
}

// removeElementFromPendingList
//
//

void IOFWUserPHYPacketListener::removeElementFromPendingList( PHYRxElement * element )
{
	DebugLog( "IOFWUserPHYPacketListener::removeElementFromPendingList - element = %p\n", element );

	// element on the pending list should not be in the free state
	
	FWKLOGASSERT( element->state != kFreeState );

	// remove from pending list
	
	//
	// handle head / next ptr
	//
	
	if( fPendingListHead == element )
	{
		FWKLOGASSERT( element->prev == NULL );
	
		fPendingListHead = element->next;
		if( fPendingListHead != NULL )
		{
			fPendingListHead->prev = NULL;
		}		
	}
	else
	{
		FWPANICASSERT( element->prev != NULL );
		
		element->prev->next = element->next;		
	}
	
	//
	// handle tail / previous ptr
	//
	
	if( fPendingListTail == element )
	{
		FWKLOGASSERT( element->next == NULL );
	
		fPendingListTail = element->prev;
		if( fPendingListTail != NULL )
		{
			fPendingListTail->prev = NULL;
		}
	}
	else
	{
		FWPANICASSERT( element->next != NULL );		
	
		element->next->prev = element->prev;
	}
	
	//
	// reset link ptrs
	// 
	
	element->next = NULL;
	element->prev = NULL;
}
