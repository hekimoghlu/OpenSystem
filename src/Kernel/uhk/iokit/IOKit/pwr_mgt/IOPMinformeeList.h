/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 20, 2024.
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
#include <libkern/c++/OSObject.h>
#include <IOKit/IOLocks.h>
#include <IOKit/IOReturn.h>

class IOPMinformee;
class IOService;
extern uint32_t gSleepAckTimeout;

class IOPMinformeeList : public OSObject
{
	OSDeclareDefaultStructors(IOPMinformeeList);
	friend class IOPMinformee;

private:
// pointer to first informee in the list
	IOPMinformee       *firstItem;
// how many informees are in the list
	unsigned long       length;

public:
	void initialize( void );
	void free( void ) APPLE_KEXT_OVERRIDE;

	unsigned long numberOfItems( void );

	LIBKERN_RETURNS_NOT_RETAINED IOPMinformee *appendNewInformee( IOService * newObject );

// OBSOLETE
// do not use addToList(); Use appendNewInformee() instead
	IOReturn addToList(LIBKERN_CONSUMED IOPMinformee *   newInformee );
	IOReturn removeFromList( IOService * theItem );

	LIBKERN_RETURNS_NOT_RETAINED IOPMinformee * firstInList( void );
	LIBKERN_RETURNS_NOT_RETAINED IOPMinformee * nextInList( IOPMinformee * currentItem );

	LIBKERN_RETURNS_NOT_RETAINED IOPMinformee * findItem( IOService * driverOrChild );

// This lock must be held while modifying list or length
	static IORecursiveLock * getSharedRecursiveLock( void );
};
