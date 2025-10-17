/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 20, 2024.
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
 */
#ifndef _SECURITY_REFCOUNT_H_
#define _SECURITY_REFCOUNT_H_

#include <security_utilities/threading.h>
#include <libkern/OSAtomic.h>

namespace Security {


//
// RefCount/RefPointer - a simple reference counting facility.
//
// To make an object reference-counted, inherit from RefCount. To track refcounted
// objects, use RefPointer<TheType>, where TheType must inherit from RefCount.
//
// RefCount is thread safe - any number of threads can hold and manipulate references
// in parallel. It does however NOT protect the contents of your object - just the
// reference count itself. If you need to share your object contents, you must provide
// appropriate locking yourself.
//
// There is no (thread safe) way to determine whether you are the only thread holding
// a pointer to a particular RefCount object. Thus there is no (thread safe)
// way to "demand copy" a RefCount subclass. Trust me; it's been tried. Don't.
//

// Uncomment to debug refcounts
//# define DEBUG_REFCOUNTS 1

#if DEBUG_REFCOUNTS
# define RCDEBUG_CREATE()   secinfo("refcount", "%p: CREATE", this)
# define RCDEBUG(_kind, n)  secinfo("refcount", "%p: %s: %d", this, #_kind, n)
#else
# define RCDEBUG_CREATE()         /* nothing */
# define RCDEBUG(kind, _args...)  /* nothing */
#endif


//
// Base class for reference counted objects
//
class RefCount {	
public:
	RefCount() : mRefCount(0) { RCDEBUG_CREATE(); }

protected:
	template <class T> friend class RefPointer;

	void ref() const
    {
        OSAtomicIncrement32(&mRefCount);
        RCDEBUG(UP, mRefCount);
    }
	
    unsigned int unref() const
    {
        RCDEBUG(DOWN, mRefCount - 1);
        return OSAtomicDecrement32(&mRefCount);
    }

private:
    volatile mutable int32_t mRefCount;
};


//
// A pointer type supported by reference counts.
// T must be derived from RefCount.
//
template <class T>
class RefPointer {
	template <class Sub> friend class RefPointer; // share with other instances
public:
	RefPointer() : ptr(0) {}			// default to NULL pointer
	RefPointer(const RefPointer& p) { if (p) p->ref(); ptr = p.ptr; }
	RefPointer(T *p) { if (p) p->ref(); ptr = p; }
	
	template <class Sub>
	RefPointer(const RefPointer<Sub>& p) { if (p) p->ref(); ptr = p.ptr; }
    
	~RefPointer() { release(); }

	RefPointer& operator = (const RefPointer& p)	{ setPointer(p.ptr); return *this; }
 	RefPointer& operator = (T * p)					{ setPointer(p); return *this; }

	template <class Sub>
	RefPointer& operator = (const RefPointer<Sub>& p) { setPointer(p.ptr); return *this; }

	// dereference operations
    T* get () const				{ _check(); return ptr; }	// mimic unique_ptr
	operator T * () const		{ _check(); return ptr; }
	T * operator -> () const	{ _check(); return ptr; }
	T & operator * () const		{ _check(); return *ptr; }

protected:
	void release_internal()
    {
        if (ptr && ptr->unref() == 0)
        {
            delete ptr;
            ptr = NULL;
        }
    }
	
    void release()
    {
        StLock<Mutex> mutexLock(mMutex);
        release_internal();
    }
    
    void setPointer(T *p)
    {
        StLock<Mutex> mutexLock(mMutex);
        if (p)
        {
            p->ref();
        }
        
        release_internal();
        ptr = p;
    }
	
	void _check() const { }

	T *ptr;
    Mutex mMutex;
};

template <class T>
bool operator <(const RefPointer<T> &r1, const RefPointer<T> &r2)
{
	T *p1 = r1.get(), *p2 = r2.get();
	return p1 && p2 ? *p1 < *p2 : p1 < p2;
}

template <class T>
bool operator ==(const RefPointer<T> &r1, const RefPointer<T> &r2)
{
	T *p1 = r1.get(), *p2 = r2.get();
	return p1 && p2 ? *p1 == *p2 : p1 == p2;
}

template <class T>
bool operator !=(const RefPointer<T> &r1, const RefPointer<T> &r2)
{
	T *p1 = r1.get(), *p2 = r2.get();
	return p1 && p2 ? *p1 != *p2 : p1 != p2;
}

} // end namespace Security

#endif // !_SECURITY_REFCOUNT_H_
