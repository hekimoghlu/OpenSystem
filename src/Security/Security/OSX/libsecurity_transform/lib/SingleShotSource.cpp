/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 8, 2023.
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


#include "SingleShotSource.h"
#include "Utilities.h"
#include <string>

using namespace std;

CFStringRef gSingleShotSourceName = CFSTR("Single Shot Source");

SingleShotSource::SingleShotSource(CFTypeRef value, Transform* t, CFStringRef name) :
	Source(gSingleShotSourceName, t, name)
{
	SetValue(value);
}

void SingleShotSource::DoActivate()
{
	// Make sure our destination doesn't vanish while we are sending it data (or the final NULL)
	CFRetainSafe(mDestination->GetCFObject());
	
	// take our value and send it on its way
	mDestination->SetAttribute(mDestinationName, GetValue());
	
	// send an end of stream
	mDestination->SetAttribute(mDestinationName, NULL);

	CFReleaseSafe(mDestination->GetCFObject());
}



Boolean SingleShotSource::Equal(const CoreFoundationObject* obj)
{
	if (Source::Equal(obj))
	{
		const SingleShotSource* sss = (const SingleShotSource*) obj;
		return CFEqual(GetValue(), sss->GetValue());
	}
	
	return false;
}



CFTypeRef SingleShotSource::Make(CFTypeRef value, Transform* t, CFStringRef name)
{
	return CoreFoundationHolder::MakeHolder(gInternalCFObjectName, new SingleShotSource(value, t, name));
}



std::string SingleShotSource::DebugDescription()
{
	string result = Source::DebugDescription() + ": SingleShotSource ";
	
	char buffer[256];
	snprintf(buffer, sizeof(buffer), "(value = %p)", GetValue());
	
	result += buffer;
	
	return result;
}
