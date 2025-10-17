/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 3, 2022.
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
/*
 *  IOFireWireUserClientIniter.h
 *  IOFireWireFamily
 *
 *  Created by NWG on Wed Jan 24 2001.
 *  Copyright (c) 2001 Apple Computer, Inc. All rights reserved.
 *
 */

#ifndef __IOKIT_IOFireWireUserClientIniter_H__
#define __IOKIT_IOFireWireUserClientIniter_H__

#include <IOKit/IOService.h>

/*! @class IOFireWireUserClientIniter
*/
class IOFireWireUserClientIniter : public IOService 
{
    OSDeclareDefaultStructors(IOFireWireUserClientIniter);

private:
	/*! @struct ExpansionData
		@discussion This structure will be used to expand the capablilties of the class in the future.
		*/    
	struct ExpansionData { };
	
	/*! @var reserved
		Reserved for future use.  (Internal use only)  */

protected:
	ExpansionData *reserved;

private:
	IOService*						fProvider;

	static IORecursiveLock *		sIniterLock;

public:
	virtual bool					start(IOService* provider) APPLE_KEXT_OVERRIDE;
	virtual bool					init(OSDictionary* propTable) APPLE_KEXT_OVERRIDE;
	virtual void					free(void) APPLE_KEXT_OVERRIDE;
	virtual void					stop(IOService* provider) APPLE_KEXT_OVERRIDE;
	
protected:
	void				mergeProperties( IORegistryEntry * dest, OSDictionary * src );
	void				mergeDictionaries( OSDictionary * dest, OSDictionary * src );
	OSDictionary*		dictionaryDeepCopy(OSDictionary* srcDictionary);

private:
    OSMetaClassDeclareReservedUnused(IOFireWireUserClientIniter, 0);
    OSMetaClassDeclareReservedUnused(IOFireWireUserClientIniter, 1);
    OSMetaClassDeclareReservedUnused(IOFireWireUserClientIniter, 2);
    OSMetaClassDeclareReservedUnused(IOFireWireUserClientIniter, 3);
};

#endif//#ifndef __IOKIT_IOFireWireUserClientIniter_H__
