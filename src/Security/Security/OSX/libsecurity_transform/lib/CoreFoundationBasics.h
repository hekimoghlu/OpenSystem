/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 18, 2024.
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

#ifndef __CORE_FOUNDATION_BASICS__
#define __CORE_FOUNDATION_BASICS__



/*
	This file contains basics for supporting our objects as CoreFoundation
	objects.
*/

#include <CoreFoundation/CFRuntime.h>
#include <string>
#include "LinkedList.h"

void CoreFoundationObjectsRegister();

extern const CFStringRef gInternalCFObjectName;
extern const CFStringRef gInternalProtectedCFObjectName;

struct CoreFoundationHolder;

class CoreFoundationObject
{
private:
	CoreFoundationHolder* mHolder;

protected:
	CoreFoundationObject(CFStringRef objectType);

protected:
	CFStringRef mObjectType;

public:
	virtual ~CoreFoundationObject();

    // compares mHolder's pointer, if you override you need
    // to also override Hash.
	virtual Boolean Equal(const CoreFoundationObject* object);
	virtual CFHashCode Hash();
	virtual std::string FormattingDescription(CFDictionaryRef options);
	virtual std::string DebugDescription();
	// default is to call delete.   Complex objects with queues may wish to release the queue and use the queue finalizer to call delete.
	virtual void Finalize();

	// register your class with the CFRuntime.  You must supply
	// a class name for your object.
	static void RegisterObject(CFStringRef name, bool protectFromDelete);
	static CFTypeID FindObjectType(CFStringRef name);
	static LinkedListHeader* GetClassRegistryList();
	
	void SetHolder(CoreFoundationHolder* holder) {mHolder = holder;}
	CFTypeRef GetCFObject() {return mHolder;}
	
	CFStringRef GetTypeAsCFString();
};



struct CoreFoundationHolder
{
	CFRuntimeBase mRuntimeBase;
	CoreFoundationObject* mObject;
	// name should really be a CFStringRef, not std::string (better memory use)
	static CoreFoundationHolder* MakeHolder(CFStringRef name, CoreFoundationObject* object);
	static CoreFoundationObject* ObjectFromCFType(CFTypeRef cfRef);
};



#endif
