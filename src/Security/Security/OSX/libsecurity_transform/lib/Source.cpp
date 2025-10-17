/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 17, 2024.
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

#include "Source.h"
#include "Utilities.h"
#include "c++utils.h"

#include <string>

using namespace std;

Source::Source(CFStringRef sourceObjectName, Transform* destination, CFStringRef destinationName) :
	CoreFoundationObject(sourceObjectName),
	mDestination(destination),
	mDestinationName(destinationName)
{
    CFStringRef queueName = CFStringCreateWithFormat(NULL, NULL, CFSTR("source:%@"), sourceObjectName);
    char *queueName_cstr = utf8(queueName);
	
	mLastValue = NULL;
	mDispatchQueue = MyDispatchQueueCreate(queueName_cstr, NULL);
    free((void*)queueName_cstr);
    CFReleaseNull(queueName);
}



Source::~Source()
{
	if (mLastValue != NULL)
	{
		CFReleaseNull(mLastValue);
	}
	
	dispatch_release(mDispatchQueue);
}



void Source::Activate()
{
	dispatch_async(mDispatchQueue, ^{DoActivate();});
}



void Source::SetValue(CFTypeRef value)
{
	if (value == mLastValue)
	{
		return;
	}
	
	// is there an existing value?  If so, release it
    CFReleaseNull(mLastValue);

    mLastValue = CFRetainSafe(value);
}



Boolean Source::Equal(const CoreFoundationObject* obj)
{
	if (CoreFoundationObject::Equal(obj))
	{
		const Source* objSource = (const Source*) obj;
		if (objSource->mDestination == mDestination &&
			CFStringCompare(objSource->mDestinationName, mDestinationName, 0) == kCFCompareEqualTo)
		{
			return true;
		}
	}
	
	return false;
}



std::string Source::DebugDescription()
{
	string result = CoreFoundationObject::DebugDescription() + ": Source ";
	
	char buffer[256];
	snprintf(buffer, sizeof(buffer), "(Destination = %p, name = %s)", mDestination, StringFromCFString(mDestinationName).c_str());
	
	result += buffer;
	
	return result;
}

