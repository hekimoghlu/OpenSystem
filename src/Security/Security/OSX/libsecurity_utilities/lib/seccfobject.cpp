/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 9, 2024.
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
#include <security_utilities/seccfobject.h>
#include <security_utilities/cfclass.h>
#include <security_utilities/errors.h>
#include <security_utilities/debugging.h>
#include <os/lock.h>

#include <list>
#include <security_utilities/globalizer.h>
#if( __cplusplus <= 201103L)
#include <stdatomic.h>
#endif

SecPointerBase::SecPointerBase(const SecPointerBase& p)
{
	if (p.ptr)
	{
		CFRetain(p.ptr->operator CFTypeRef());
	}
	ptr = p.ptr;
}


SecPointerBase::SecPointerBase(SecCFObject *p)
{
	if (p && !p->isNew())
	{
		CFRetain(p->operator CFTypeRef());
	}
	ptr = p;
}



SecPointerBase::~SecPointerBase()
{
	if (ptr)
	{
		CFRelease(ptr->operator CFTypeRef());
	}
}



SecPointerBase& SecPointerBase::operator = (const SecPointerBase& p)
{
	if (p.ptr)
	{
		CFTypeRef tr = p.ptr->operator CFTypeRef();
		CFRetain(tr);
	}
	if (ptr)
	{
		CFRelease(ptr->operator CFTypeRef());
	}
	ptr = p.ptr;
	return *this;
}



void SecPointerBase::assign(SecCFObject * p)
{
	if (p && !p->isNew())
	{
		CFRetain(p->operator CFTypeRef());
	}
	if (ptr)
	{
		CFRelease(ptr->operator CFTypeRef());
	}
	ptr = p;
}



void SecPointerBase::copy(SecCFObject * p)
{
	if (ptr)
	{
		CFRelease(ptr->operator CFTypeRef());
	}
	
	ptr = p;
}



//
// SecCFObject
//
SecCFObject *
SecCFObject::optional(CFTypeRef cfTypeRef) _NOEXCEPT
{
	if (!cfTypeRef)
		return NULL;

	return const_cast<SecCFObject *>(reinterpret_cast<const SecCFObject *>(reinterpret_cast<const uint8_t *>(cfTypeRef) + kAlignedRuntimeSize));
}

SecCFObject *
SecCFObject::required(CFTypeRef cfTypeRef, OSStatus error)
{
	SecCFObject *object = optional(cfTypeRef);
	if (!object)
		MacOSError::throwMe(error);

	return object;
}

void *
SecCFObject::allocate(size_t size, const CFClass &cfclass)
{
	CFTypeRef p = _CFRuntimeCreateInstance(NULL, cfclass.typeID,
		size + kAlignedRuntimeSize - sizeof(CFRuntimeBase), NULL);
	if (p == NULL)
		throw std::bad_alloc();

	atomic_flag_clear(&((SecRuntimeBase*) p)->isOld);

	void *q = ((u_int8_t*) p) + kAlignedRuntimeSize;

	return q;
}

void
SecCFObject::operator delete(void *object) _NOEXCEPT
{
	CFTypeRef cfType = reinterpret_cast<CFTypeRef>(reinterpret_cast<const uint8_t *>(object) - kAlignedRuntimeSize);

    CFAllocatorRef allocator = CFGetAllocator(cfType);
    CFAllocatorDeallocate(allocator, (void*) cfType);
}

SecCFObject::SecCFObject()
{
    mRetainCount = 1;
    mRetainLock = OS_UNFAIR_LOCK_INIT;
}

uint32_t SecCFObject::updateRetainCount(intptr_t direction, uint32_t *oldCount)
{
    os_unfair_lock_lock(&mRetainLock);

    if (oldCount != NULL)
    {
        *oldCount = mRetainCount;
    }
    
    if (direction != -1 || mRetainCount != 0)
    {
        // if we are decrementing
        if (direction == -1 || UINT32_MAX != mRetainCount)
        {
            mRetainCount += direction;
        }
    }
    
    uint32_t result = mRetainCount;

    os_unfair_lock_unlock(&mRetainLock);
    
    return result;
}



SecCFObject::~SecCFObject()
{
	//SECURITY_DEBUG_SEC_DESTROY(this);
}

bool
SecCFObject::equal(SecCFObject &other)
{
	return this == &other;
}

CFHashCode
SecCFObject::hash()
{
	return CFHashCode(this);
}

CFStringRef
SecCFObject::copyFormattingDesc(CFDictionaryRef dict)
{
	return NULL;
}

CFStringRef
SecCFObject::copyDebugDesc()
{
	return NULL;
}

CFTypeRef
SecCFObject::handle(bool retain) _NOEXCEPT
{
	CFTypeRef cfType = *this;
	if (retain && !isNew()) CFRetain(cfType);
	return cfType;
}



void
SecCFObject::aboutToDestruct()
{
}



Mutex*
SecCFObject::getMutexForObject() const
{
	return NULL; // we only worry about descendants of KeychainImpl and ItemImpl
}



bool SecCFObject::mayDelete()
{
    return true;
}
