/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 8, 2022.
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

#include <libkern/OSAtomic.h>

#include "TransformFactory.h"
#include "NullTransform.h"
#include "Digest.h"
#include "EncryptTransform.h"
#include "GroupTransform.h"
#include "Utilities.h"


void TransformFactory::RegisterTransforms()
{
	RegisterTransform_prelocked(NullTransform::MakeTransformFactory(), NULL);
	RegisterTransform_prelocked(DigestTransform::MakeTransformFactory(), NULL);
	RegisterTransform_prelocked(EncryptTransform::MakeTransformFactory(), NULL);
	RegisterTransform_prelocked(DecryptTransform::MakeTransformFactory(), NULL);
	RegisterTransform_prelocked(GroupTransform::MakeTransformFactory(), NULL);
}

CFMutableDictionaryRef TransformFactory::gRegistered;
dispatch_once_t TransformFactory::gSetup;
dispatch_queue_t TransformFactory::gRegisteredQueue;

bool TransformFactory::RegisterTransform_prelocked(TransformFactory* tf, CFStringRef cfname)
{
    if (!CFDictionaryContainsKey(gRegistered, tf->mCFType)) {
        CFDictionaryAddValue(gRegistered, tf->mCFType, tf);
        if (!cfname) {
            CoreFoundationObject::RegisterObject(tf->mCFType, false);
        } else {
            if (!CoreFoundationObject::FindObjectType(cfname)) {
                CoreFoundationObject::RegisterObject(cfname, false);
            }
        }
    }
    
    return true;
}


void TransformFactory::RegisterTransform(TransformFactory* tf, CFStringRef cfname)
{
    dispatch_once_f(&gSetup, NULL, Setup);
    dispatch_barrier_sync(gRegisteredQueue, ^{
        RegisterTransform_prelocked(tf, cfname);
    });
}

void TransformFactory::Setup(void *)
{
    gRegisteredQueue = dispatch_queue_create("com.apple.security.TransformFactory.Registered", DISPATCH_QUEUE_CONCURRENT);
    gRegistered = CFDictionaryCreateMutable(NULL, 0, &kCFCopyStringDictionaryKeyCallBacks, NULL);
    RegisterTransforms();
}

void TransformFactory::Setup()
{
    dispatch_once_f(&gSetup, NULL, Setup);
}

TransformFactory* TransformFactory::FindTransformFactoryByType(CFStringRef name)
{
    dispatch_once_f(&gSetup, NULL, Setup);
    __block TransformFactory *ret;
    dispatch_barrier_sync(gRegisteredQueue, ^{
        ret = (TransformFactory*)CFDictionaryGetValue(gRegistered, name);
    });
    return ret;
}



SecTransformRef TransformFactory::MakeTransformWithType(CFStringRef type, CFErrorRef* baseError) CF_RETURNS_RETAINED
{
	TransformFactory* tf = FindTransformFactoryByType(type);
	if (!tf)
	{
		if (baseError != NULL)
		{
#if 0
            // This version lists out all regestered transform types.
            // It is useful more for debugging then for anything else,
            // so it is great to keep around, but normally not so good
            // to run.
			dispatch_barrier_sync(gRegisteredQueue, ^(void) {
                CFMutableStringRef transformNames = CFStringCreateMutable(NULL, 0);
                CFIndex numberRegistered = CFDictionaryGetCount(gRegistered);
                CFStringRef *names = (CFStringRef*)malloc(numberRegistered * sizeof(CFStringRef));
                if (names == NULL) {
                    *baseError = CreateSecTransformErrorRef(errSecMemoryError,
                                                            "The %s transform names can't be allocated.", type);
                    return NULL;
                }

                CFDictionaryGetKeysAndValues(gRegistered, (const void**)names, NULL);
                for(int i = 0; i < numberRegistered; i++) {
                    if (i != 0) {
                        CFStringAppend(transformNames, CFSTR(", "));
                    }
                    CFStringAppend(transformNames, names[i]);
                }
                
                free(names);

                *baseError = CreateSecTransformErrorRef(kSecTransformTransformIsNotRegistered, 
                                                        "The %s transform is not registered, choose from: %@", type,transformNames);

            });
#else
            *baseError = CreateSecTransformErrorRef(kSecTransformTransformIsNotRegistered, 
                                                    CFSTR("The %@ transform is not registered"), type);
#endif
		}
		
		return NULL;
	}
	else
	{
		return tf->Make();
	}
}



TransformFactory::TransformFactory(CFStringRef type, bool registerGlobally, CFStringRef cftype) : mCFType(type)
{
	if (registerGlobally)
	{
		CoreFoundationObject::RegisterObject(cftype ? cftype : type, false);
	}
}
