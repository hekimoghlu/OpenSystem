/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 23, 2025.
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
#ifndef _IOFWPHYPACKETLISTENER_H_
#define _IOFWPHYPACKETLISTENER_H_

#include <libkern/c++/OSObject.h>

class IOFireWireController;

// Callback when phy packet is received
typedef void (*FWPHYPacketCallback)( void *refcon, UInt32 data1, UInt32 data2 );

/*! @class IOFWPHYPacketListener
*/

class IOFWPHYPacketListener : public OSObject
{
	OSDeclareDefaultStructors( IOFWPHYPacketListener );

	friend class IOFireWireController;
	
protected:
	
	IOFireWireController *		fControl;
	FWPHYPacketCallback			fCallback;
	void *						fRefCon;

	static IOFWPHYPacketListener * createWithController( IOFireWireController * controller );

    virtual bool initWithController( IOFireWireController * control );
    virtual void free( void ) APPLE_KEXT_OVERRIDE;

public:

	virtual IOReturn activate( void );
	virtual void deactivate( void );

	virtual void setCallback( FWPHYPacketCallback callback );
	virtual void setRefCon( void * refcon );
	virtual void * getRefCon( void );

protected:
	virtual void processPHYPacket( UInt32 data1, UInt32 data2 );

    OSMetaClassDeclareReservedUnused( IOFWPHYPacketListener, 0 );
    OSMetaClassDeclareReservedUnused( IOFWPHYPacketListener, 1 );
    OSMetaClassDeclareReservedUnused( IOFWPHYPacketListener, 2 );
    OSMetaClassDeclareReservedUnused( IOFWPHYPacketListener, 3 );
    OSMetaClassDeclareReservedUnused( IOFWPHYPacketListener, 4 );
    OSMetaClassDeclareReservedUnused( IOFWPHYPacketListener, 5 );
    OSMetaClassDeclareReservedUnused( IOFWPHYPacketListener, 6 );
    OSMetaClassDeclareReservedUnused( IOFWPHYPacketListener, 7 );
    OSMetaClassDeclareReservedUnused( IOFWPHYPacketListener, 8 );
    OSMetaClassDeclareReservedUnused( IOFWPHYPacketListener, 9 );
};

#endif
