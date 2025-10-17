/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 10, 2024.
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

#ifndef __TRANSFORM_FACTORY__
#define __TRANSFORM_FACTORY__

#include "Transform.h"
#include "LinkedList.h"

class TransformFactory
{
protected:
	static void Register(TransformFactory* tf);
    static dispatch_once_t gSetup;
    static dispatch_queue_t gRegisteredQueue;
    static CFMutableDictionaryRef gRegistered;
    
	CFStringRef mCFType;

	static TransformFactory* FindTransformFactoryByType(CFStringRef type);
	static void RegisterTransforms();
	static void RegisterTransform(TransformFactory* tf, CFStringRef cfname = NULL);
    static void Setup(void *);

private:
    static bool RegisterTransform_prelocked(TransformFactory* tf, CFStringRef name);

public:
	static SecTransformRef MakeTransformWithType(CFStringRef type, CFErrorRef* baseError) CF_RETURNS_RETAINED;

	TransformFactory(CFStringRef type, bool registerGlobally = false, CFStringRef cftype = NULL);
	static void Setup();
	virtual CFTypeRef Make() = 0;
    CFStringRef GetTypename() { return mCFType; };
};



#endif
