/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 1, 2024.
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
#ifndef _IOKIT_IOFIREWIREMULTIISOCHRECEIVE_H_
#define _IOKIT_IOFIREWIREMULTIISOCHRECEIVE_H_

class IOFireWireMultiIsochReceiveListener;
class IOFireWireMultiIsochReceivePacket;
class IOFireWireController;

typedef IOReturn (*FWMultiIsochReceiveListenerCallback)(void *refcon, IOFireWireMultiIsochReceivePacket *pPacket);

// These are the parameters clients can set which help us to optimize the mult-isoch-receiver 
// polling interval, and memory resources 
typedef struct FWMultiIsochReceiveListenerParamsStruct
	{
		// How much latency, from when the packet arrives to when the client is notified, can the client tolerate. 
		UInt32 maxLatencyInFireWireCycles;
		
		// In bits per second, the expected bit-rate of the incoming stream
		UInt32 expectedStreamBitRate;
		
		// How long does the client expect to hold onto packets objects before returning them back to the receiver
		UInt32 clientPacketReturnLatencyInFireWireCycles;
	}FWMultiIsochReceiveListenerParams;

/*! @class IOFireWireMultiIsochReceiveListener
*/

class IOFireWireMultiIsochReceiveListener : public OSObject
	{
		friend class IOFireWireLink;
		
	protected:
		OSDeclareDefaultStructors(IOFireWireMultiIsochReceiveListener)
		bool init(IOFireWireController *fwController,
				  UInt32 receiveChannel,
				  FWMultiIsochReceiveListenerCallback callback,
				  void *pCallbackRefCon,
				  FWMultiIsochReceiveListenerParams *pListenerParams);
		void free() APPLE_KEXT_OVERRIDE;
	public:
		static IOFireWireMultiIsochReceiveListener *create(IOFireWireController *fwController,
														   UInt32 channel,
														   FWMultiIsochReceiveListenerCallback callback,
														   void *pCallbackRefCon,
														   FWMultiIsochReceiveListenerParams *pListenerParams);
		
		// Call this to activate the listener
		IOReturn Activate();
		
		// Call this to deactivate the listener
		IOReturn Deactivate();
		
		// Call this to modify the callback/refcon pointers. Only call this when not activated!
		IOReturn SetCallback(FWMultiIsochReceiveListenerCallback callback,
							 void *pCallbackRefCon);
		
		// Accessors
		inline UInt32 getReceiveChannel(void) {return fChannel;};
		inline FWMultiIsochReceiveListenerCallback getCallback(void){return fClientCallback;}; 
		inline void * getRefCon(void){return fClientCallbackRefCon;};
		inline bool getActivatedState(void) {return fActivated;};
		
	protected:
		IOFireWireController *fControl;
		UInt32 fChannel;
		FWMultiIsochReceiveListenerCallback fClientCallback;
		void *fClientCallbackRefCon;
		bool fActivated;
		FWMultiIsochReceiveListenerParams *fListenerParams;
	};

#define kMaxRangesPerMultiIsochReceivePacket 6

/*! @class IOFireWireMultiIsochReceivePacket
*/

class IOFireWireMultiIsochReceivePacket : public OSObject
	{
		OSDeclareDefaultStructors(IOFireWireMultiIsochReceivePacket)
		bool init(IOFireWireController *fwController);
		void free() APPLE_KEXT_OVERRIDE;
	public:
		static IOFireWireMultiIsochReceivePacket *create(IOFireWireController *fwController);
		
		// The clients who are passed this packet by the 
		// multi-isoch receiver calling their callback
		// MUST call clientDone() on this packet to
		// return it back for reuse!
		void clientDone(void);
		
		UInt32 isochChannel(void);
		UInt32 packetReceiveTime(void);
		
		UInt32 isochPayloadSize(void);	// The size of just the isoch payload, not including header/trailer quads.
		inline UInt32 isochPacketSize(void) {return isochPayloadSize()+8; };	// The size of the packet, including header/trailer quads.
		
		// This returns a memory descriptor to the client. The client must call complete(), and release() on the
		// memory descriptor when done.
		IOMemoryDescriptor *createMemoryDescriptorForRanges(void);
		
		// These should be treated as read-only by clients,
		// as should the data contained in these buffers!
		IOAddressRange ranges[kMaxRangesPerMultiIsochReceivePacket] ;
		UInt32 numRanges;
		
		// These should be treated private for clients!
		// Messing with them will screw up the bookkeepping
		// in the Multi-Isoch Receiver!
		UInt32 numClientReferences;
		void* elements[kMaxRangesPerMultiIsochReceivePacket];
		
	protected:
		IOFireWireController *fControl;
	};

#endif
