/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 2, 2023.
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
#ifndef _IOKIT_IOFWSBP2PSEUDOADDRESSSPACE_H
#define _IOKIT_IOFWSBP2PSEUDOADDRESSSPACE_H

#include <IOKit/firewire/IOFWPseudoAddressSpace.h>

#pragma mark -

class IOFireWireUnit;

/*! 
	@class IOFWSBP2PseudoAddressSpace
*/

class IOFWSBP2PseudoAddressSpace : public IOFWPseudoAddressSpace
{
    OSDeclareDefaultStructors(IOFWSBP2PseudoAddressSpace)
	
protected:
	
	/*! 
		@struct ExpansionData
		@discussion This structure will be used to expand the capablilties of the class in the future.
    */  
	  
    struct ExpansionData { };

	/*! 
		@var reserved
		Reserved for future use.  (Internal use only)  
	*/
    
	ExpansionData *reserved;

public:

	virtual void setAddressLo( UInt32 addressLo );

	static IOFWSBP2PseudoAddressSpace * simpleRead(	IOFireWireBus *	control,
													FWAddress *		addr, 
													UInt32 			len, 
													const void *	data );

	static IOFWSBP2PseudoAddressSpace * simpleRW(	IOFireWireBus *	control,
													FWAddress *		addr, 
													UInt32 			len, 
													void *			data );
																											
	static IOFWSBP2PseudoAddressSpace * createPseudoAddressSpace(	IOFireWireBus * control,
																	IOFireWireUnit * unit,
																	FWAddress *		addr, 
																	UInt32 			len, 
																	FWReadCallback 	reader, 
																	FWWriteCallback	writer, 
																	void *			refcon );
															
private:
    OSMetaClassDeclareReservedUnused(IOFWSBP2PseudoAddressSpace, 0);
    OSMetaClassDeclareReservedUnused(IOFWSBP2PseudoAddressSpace, 1);
    OSMetaClassDeclareReservedUnused(IOFWSBP2PseudoAddressSpace, 2);
    OSMetaClassDeclareReservedUnused(IOFWSBP2PseudoAddressSpace, 3);
    OSMetaClassDeclareReservedUnused(IOFWSBP2PseudoAddressSpace, 4);
    OSMetaClassDeclareReservedUnused(IOFWSBP2PseudoAddressSpace, 5);
    OSMetaClassDeclareReservedUnused(IOFWSBP2PseudoAddressSpace, 6);
    OSMetaClassDeclareReservedUnused(IOFWSBP2PseudoAddressSpace, 7);
    OSMetaClassDeclareReservedUnused(IOFWSBP2PseudoAddressSpace, 8);
    OSMetaClassDeclareReservedUnused(IOFWSBP2PseudoAddressSpace, 9);
	
};

#endif
