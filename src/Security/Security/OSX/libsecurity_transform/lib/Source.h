/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 11, 2024.
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

#ifndef __SOURCE__
#define __SOURCE__



#include "SecTransform.h"
#include "CoreFoundationBasics.h"
#include "Transform.h"

class Source : public CoreFoundationObject
{
protected:
	Transform* mDestination;
	CFStringRef mDestinationName;
	CFTypeRef mLastValue;
	dispatch_queue_t mDispatchQueue;

	void SetValue(CFTypeRef value);

	Source(CFStringRef sourceObjectName, Transform* destination, CFStringRef destinationName);

public:
	virtual ~Source();
	
	void Activate();
	virtual void DoActivate() = 0;
	
	Boolean Equal(const CoreFoundationObject* obj);
	CFTypeRef GetValue() const {return mLastValue;}
	std::string DebugDescription();
};

#endif
