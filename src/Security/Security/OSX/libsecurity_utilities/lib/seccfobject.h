/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 13, 2022.
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
#ifndef _SECCFOBJECT_H
#define _SECCFOBJECT_H

#include <CoreFoundation/CFRuntime.h>
#include <new>
#include "threading.h"
#include <os/lock.h>

#if( __cplusplus <= 201103L)
#include <stdatomic.h>
#else
#include <atomic>
#endif

namespace Security {

class CFClass;

#define SECCFFUNCTIONS_BASE(OBJTYPE, APIPTR) \
\
operator APIPTR() const \
{ return (APIPTR)(this->operator CFTypeRef()); } \
\
OBJTYPE *retain() \
{ SecCFObject::handle(true); return this; } \
APIPTR CF_RETURNS_RETAINED handle() \
{ return (APIPTR)SecCFObject::handle(true); } \
APIPTR handle(bool retain) \
{ return (APIPTR)SecCFObject::handle(retain); }

#define SECCFFUNCTIONS_CREATABLE(OBJTYPE, APIPTR, CFCLASS) \
SECCFFUNCTIONS_BASE(OBJTYPE, APIPTR)\
\
void *operator new(size_t size)\
{ return SecCFObject::allocate(size, CFCLASS); }

#define SECCFFUNCTIONS(OBJTYPE, APIPTR, ERRCODE, CFCLASS) \
SECCFFUNCTIONS_CREATABLE(OBJTYPE, APIPTR, CFCLASS) \
\
static OBJTYPE *required(APIPTR ptr) \
{ if (OBJTYPE *p = dynamic_cast<OBJTYPE *>(SecCFObject::required(ptr, ERRCODE))) \
	return p; else MacOSError::throwMe(ERRCODE); } \
\
static OBJTYPE *optional(APIPTR ptr) \
{ if (SecCFObject *p = SecCFObject::optional(ptr)) \
	if (OBJTYPE *pp = dynamic_cast<OBJTYPE *>(p)) return pp; else MacOSError::throwMe(ERRCODE); \
  else return NULL; }

#define SECALIGNUP(SIZE, ALIGNMENT) (((SIZE - 1) & ~(ALIGNMENT - 1)) + ALIGNMENT)

struct SecRuntimeBase: CFRuntimeBase
{
	atomic_flag isOld;
};

class SecCFObject
{
private:
	void *operator new(size_t);

	// Align up to a multiple of 16 bytes
	static const size_t kAlignedRuntimeSize = SECALIGNUP(sizeof(SecRuntimeBase), 4);

    uint32_t mRetainCount;
    os_unfair_lock mRetainLock;

public:
	// For use by SecPointer only. Returns true once the first time it's called after the object has been created.
	bool isNew()
	{
		SecRuntimeBase *base = reinterpret_cast<SecRuntimeBase *>(reinterpret_cast<uint8_t *>(this) - kAlignedRuntimeSize);

        // atomic flags start clear, and like to go high.
        return !atomic_flag_test_and_set(&(base->isOld));
	}

	static SecCFObject *optional(CFTypeRef) _NOEXCEPT;
	static SecCFObject *required(CFTypeRef, OSStatus error);
	static void *allocate(size_t size, const CFClass &cfclass);

    SecCFObject();
	virtual ~SecCFObject();
    uint32_t updateRetainCount(intptr_t direction, uint32_t *oldCount);
    uint32_t getRetainCount() {return updateRetainCount(0, NULL);}

	static void operator delete(void *object) _NOEXCEPT;
	virtual operator CFTypeRef() const _NOEXCEPT
	{
		return reinterpret_cast<CFTypeRef>(reinterpret_cast<const uint8_t *>(this) - kAlignedRuntimeSize);
	}

	// This bumps up the retainCount by 1, by calling CFRetain(), iff retain is true
	CFTypeRef handle(bool retain = true) _NOEXCEPT;

    virtual bool equal(SecCFObject &other);
    virtual CFHashCode hash();
	virtual CFStringRef copyFormattingDesc(CFDictionaryRef dict);
	virtual CFStringRef copyDebugDesc();
	virtual void aboutToDestruct();
	virtual Mutex* getMutexForObject() const;
    virtual bool mayDelete();
};

//
// A pointer type for SecCFObjects.
// T must be derived from SecCFObject.
//
class SecPointerBase
{
public:
	SecPointerBase() : ptr(NULL)
	{}
	SecPointerBase(const SecPointerBase& p);
	SecPointerBase(SecCFObject *p);
	~SecPointerBase();
	SecPointerBase& operator = (const SecPointerBase& p);

protected:
 	void assign(SecCFObject * p);
	void copy(SecCFObject * p);
	SecCFObject *ptr;
};

template <class T>
class SecPointer : public SecPointerBase
{
public:
	SecPointer() : SecPointerBase() {}
	SecPointer(const SecPointer& p) : SecPointerBase(p) {}
	SecPointer(T *p): SecPointerBase(p) {}
	SecPointer &operator =(T *p) { this->assign(p); return *this; }
	SecPointer &take(T *p) { this->copy(p); return *this; }
	T *yield() { T *result = static_cast<T *>(ptr); ptr = NULL; return result; }
	
	// dereference operations
    T* get () const				{ return static_cast<T*>(ptr); }	// mimic unique_ptr
	operator T * () const		{ return static_cast<T*>(ptr); }
	T * operator -> () const	{ return static_cast<T*>(ptr); }
	T & operator * () const		{ return *static_cast<T*>(ptr); }

    SecPointer& operator=(const SecPointer& other) { SecPointerBase::operator=(other); return *this; }
};

template <class T>
bool operator <(const SecPointer<T> &r1, const SecPointer<T> &r2)
{
	T *p1 = r1.get(), *p2 = r2.get();
	return p1 && p2 ? *p1 < *p2 : p1 < p2;
}

template <class T>
bool operator ==(const SecPointer<T> &r1, const SecPointer<T> &r2)
{
	T *p1 = r1.get(), *p2 = r2.get();
	return p1 && p2 ? *p1 == *p2 : p1 == p2;
}

template <class T>
bool operator !=(const SecPointer<T> &r1, const SecPointer<T> &r2)
{
	T *p1 = r1.get(), *p2 = r2.get();
	return p1 && p2 ? *p1 != *p2 : p1 != p2;
}

} // end namespace Security


#endif
