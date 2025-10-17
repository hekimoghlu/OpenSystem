/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 2, 2022.
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
#include <security_utilities/cfclass.h>
#include <security_utilities/seccfobject.h>
#include <security_utilities/threading.h>
#include <CoreFoundation/CFString.h>
#include <sys/time.h>

//
// CFClass
//
CFClass::CFClass(const char *name)
{
	// initialize the CFRuntimeClass structure
	version = 0;
	className = name;
	init = NULL;
	copy = NULL;
	finalize = finalizeType;
	equal = equalType;
	hash = hashType;
	copyFormattingDesc = copyFormattingDescType;
	copyDebugDesc = copyDebugDescType;
	
    // update because we are now doing our own reference counting
    version |= _kCFRuntimeCustomRefCount; // see ma, no hands!
    refcount = refCountForType;

	// register
	typeID = _CFRuntimeRegisterClass(this);
	assert(typeID != _kCFRuntimeNotATypeID);
}

uint32_t
CFClass::cleanupObject(intptr_t op, CFTypeRef cf, bool &zap)
{
    // the default is to not throw away the object
    zap = false;
    
    uint32_t currentCount;
    SecCFObject *obj = SecCFObject::optional(cf);

    uint32_t oldCount;
    currentCount = obj->updateRetainCount(op, &oldCount);

    if (op == 0)
    {
        return currentCount;
    }
    else if (currentCount == 0)
    {
        // we may not be able to delete if the caller has active children
        if (obj->mayDelete())
        {
            finalizeType(cf);
            zap = true; // ask the caller to release the mutex and zap the object
            return 0;
        }
        else
        {
            return currentCount;
        }
    }
    else 
    {
        return 0;
    }
}

uint32_t
CFClass::refCountForType(intptr_t op, CFTypeRef cf) _NOEXCEPT
{
    uint32_t result = 0;
    bool zap = false;

    try
    {
        SecCFObject *obj = SecCFObject::optional(cf);
		Mutex* mutex = obj->getMutexForObject();
		if (mutex == NULL)
		{
			// if the object didn't have a mutex, it wasn't cached.
			// Just clean it up and get out.
            result = cleanupObject(op, cf, zap);
		}
		else
        {
            // we have a mutex, so we need to do our cleanup operation under its control
            StLock<Mutex> _(*mutex);
            result = cleanupObject(op, cf, zap);
        }
        
        if (zap) // did we release the object?
        {
            delete obj; // should call the overloaded delete for the object
        }
    }
    catch (...)
    {
    }
    
    // keep the compiler happy
    return result;
}



void
CFClass::finalizeType(CFTypeRef cf) _NOEXCEPT
{
    /*
        We need to control the lifetime of the object.  This means
        that the cache lock has to be asserted while we are determining if the
        object should live or die.  The mutex is recursive, which means that
        we won't end up with mutex inversion.
    */
    
    SecCFObject *obj = SecCFObject::optional(cf);

    try
	{
		Mutex* mutex = obj->getMutexForObject();
		if (mutex == NULL)
		{
			// if the object didn't have a mutex, it wasn't cached.
			// Just clean it up and get out.
			obj->aboutToDestruct(); // removes the object from its associated cache.
		}
		else
        {
            StLock<Mutex> _(*mutex);
            
            if (obj->isNew())
            {
                // New objects aren't in the cache.
                // Just clean it up and get out.
                obj->aboutToDestruct(); // removes the object from its associated cache.
                return;
            }
            
            obj->aboutToDestruct(); // removes the object from its associated cache.
        }
	}
	catch(...)
	{
	}
}

Boolean
CFClass::equalType(CFTypeRef cf1, CFTypeRef cf2) _NOEXCEPT
{
	// CF checks for pointer equality and ensures type equality already
	try {
		return SecCFObject::optional(cf1)->equal(*SecCFObject::optional(cf2));
	} catch (...) {
		return false;
	}
}

CFHashCode
CFClass::hashType(CFTypeRef cf) _NOEXCEPT
{
	try {
		return SecCFObject::optional(cf)->hash();
	} catch (...) {
		return 666; /* Beasty return for error */
	}
}

CFStringRef
CFClass::copyFormattingDescType(CFTypeRef cf, CFDictionaryRef dict) _NOEXCEPT
{
	try {
		return SecCFObject::optional(cf)->copyFormattingDesc(dict);
	} catch (...) {
		return CFSTR("Exception thrown trying to format object");
	}
}

CFStringRef
CFClass::copyDebugDescType(CFTypeRef cf) _NOEXCEPT
{
	try {
		return SecCFObject::optional(cf)->copyDebugDesc();
	} catch (...) {
		return CFSTR("Exception thrown trying to format object");
	}
}


