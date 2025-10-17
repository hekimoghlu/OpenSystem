/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 2, 2025.
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
#ifndef _IOKIT_IOFWUSERPHYPACKETLISENER_H_
#define _IOKIT_IOFWUSERPHYPACKETLISENER_H_

// private
#import "IOFireWireUserClient.h"

//public
#import <IOKit/firewire/IOFWPHYPacketListener.h>

#pragma mark -

class IOFWUserPHYPacketListener : public IOFWPHYPacketListener
{
	OSDeclareDefaultStructors( IOFWUserPHYPacketListener );

	protected:

	enum ElementState 
	{
        kFreeState = 0,
		kPendingState = 1
    };
	
	enum ElementType 
	{
		kTypeNone = 0,
		kTypeData = 1,
		kTypeSkipped = 2
	};
	
	class PHYRxElement
	{
			
	public:

		PHYRxElement * 		next;			
		PHYRxElement * 		prev;
		
		ElementState		state;
		ElementType			type;
		
		UInt32				data1;
		UInt32				data2;
				
		inline UInt32 getSkippedCount( void )
		{
			return data1;
		}
		
		inline void setSkippedCount( UInt32 count )
		{
			data1 = count;
		}
	};
		
	protected:
		IOFireWireUserClient *		fUserClient;
		UInt32						fMaxQueueCount;
		UInt32						fAllocatedQueueCount;

		OSAsyncReference64			fCallbackAsyncRef;
		OSAsyncReference64			fSkippedAsyncRef;

		PHYRxElement *				fElementWaitingCompletion;
		
		PHYRxElement *				fFreeListHead;				// pointer to first free element
		PHYRxElement *				fFreeListTail;				// pointer to last free element
		
		PHYRxElement *				fPendingListHead;			// pointer to the oldest active element
		PHYRxElement *				fPendingListTail;			// pointer to newest active element

		IOLock *					fLock;
						
	public:
	
		// factory
		static IOFWUserPHYPacketListener *		withUserClient( IOFireWireUserClient * inUserClient, UInt32 queue_count );
		
		// ctor/dtor
		virtual bool	initWithUserClient( IOFireWireUserClient * inUserClient, UInt32 queue_count );
		virtual void	free( void ) APPLE_KEXT_OVERRIDE;
	
		static void		exporterCleanup( const OSObject * self );
		
		IOReturn	setPacketCallback(	OSAsyncReference64		async_ref,
										mach_vm_address_t		callback,
										io_user_reference_t		refCon );
										
		IOReturn	setSkippedCallback(	OSAsyncReference64		async_ref,
										mach_vm_address_t		callback,
										io_user_reference_t		refCon );
																			
		void		clientCommandIsComplete( FWClientCommandID commandID );

	protected:
		virtual		void processPHYPacket( UInt32 data1, UInt32 data2 ) APPLE_KEXT_OVERRIDE;

		void			sendPacketNotification( IOFWUserPHYPacketListener::PHYRxElement * element );

		IOReturn		createAllCommandElements( void );
		void			destroyAllElements( void );
		
		PHYRxElement *	allocateElement( void );
		PHYRxElement *	allocateDataElement( void );
		
		void			deallocateElement( PHYRxElement * element );
		
		void			addElementToPendingList( PHYRxElement * element );
		void			removeElementFromPendingList( PHYRxElement * element );

};

#endif
