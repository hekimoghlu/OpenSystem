/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 27, 2023.
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
//
// alloc - abstract malloc-like allocator abstraction
//
#ifndef _H_ALLOC
#define _H_ALLOC

#include <security_utilities/utilities.h>
#include <cstring>

namespace Security
{


//
// An abstract allocator superclass, based on the simple malloc/realloc/free paradigm
// that CDSA loves so much. If you have an allocation strategy and want objects
// to be allocated through it, inherit from this.
//
class Allocator {
public:
	virtual ~Allocator();
	virtual void *malloc(size_t)= 0;
	virtual void free(void *) _NOEXCEPT = 0;
	virtual void *realloc(void *, size_t)= 0;

	//
	// Template versions for added expressiveness.
	// Note that the integers are element counts, not byte sizes.
	//
	template <class T> T *alloc()
	{ return reinterpret_cast<T *>(malloc(sizeof(T))); }

	template <class T> T *alloc(UInt32 count)
	{
        size_t bytes = 0;
        if (__builtin_mul_overflow(sizeof(T), count, &bytes)) {
            throw std::bad_alloc();
        }
        return reinterpret_cast<T *>(malloc(bytes));

    }

	template <class T> T *alloc(T *old, UInt32 count)
	{
        size_t bytes = 0;
        if (__builtin_mul_overflow(sizeof(T), count, &bytes)) {
            throw std::bad_alloc();
        }
        return reinterpret_cast<T *>(realloc(old, bytes));
    }
	
        
	//
	// Happier malloc/realloc for any type. Note that these still have
	// the original (byte-sized) argument profile.
	//
	template <class T> T *malloc(size_t size)
	{ return reinterpret_cast<T *>(malloc(size)); }
	
	template <class T> T *realloc(void *addr, size_t size)
	{ return reinterpret_cast<T *>(realloc(addr, size)); }

	// All right, if you *really* have to have calloc...
	void *calloc(size_t size, size_t count)
	{
        size_t bytes = 0;
        if(__builtin_mul_overflow(size, count, &bytes)) {
            // Multiplication overflowed.
            throw std::bad_alloc();
        }
		void *addr = malloc(bytes);
		memset(addr, 0, bytes);
		return addr;
	}
	
	// compare Allocators for identity
	virtual bool operator == (const Allocator &alloc) const _NOEXCEPT;

public:
	// allocator chooser options
	enum {
		normal = 0x0000,
		sensitive = 0x0001
	};

	static Allocator &standard(UInt32 request = normal);
};


//
// You'd think that this is operator delete(const T *, Allocator &), but you'd
// be wrong. Specialized operator delete is only called during constructor cleanup.
// Use this to cleanly destroy things.
//
template <class T>
inline void destroy(T *obj, Allocator &alloc) _NOEXCEPT
{
	obj->~T();
	alloc.free(obj);
}

// untyped (release memory only, no destructor call)
inline void destroy(void *obj, Allocator &alloc) _NOEXCEPT
{
	alloc.free(obj);
}


//
// A mixin class to automagically manage your allocator.
// To allow allocation (of your object) from any instance of Allocator,
// inherit from CssmHeap. Your users can then create heap instances of your thing by
//		new (an-allocator) YourClass(...)
// or (still)
//		new YourClass(...)
// for the default allocation source. The beauty is that when someone does a
//		delete pointer-to-your-instance
// then the magic fairies will find the allocator that created the object and ask it
// to free the memory (by calling its free() method).
// The price of all that glory is memory overhead - typically one pointer per object.
//
class CssmHeap {
public:    
	void *operator new (size_t size, Allocator *alloc = NULL);
	void operator delete (void *addr, size_t size) _NOEXCEPT;
	void operator delete (void *addr, size_t size, Allocator *alloc) _NOEXCEPT;
};


//
// Here is a version of unique_ptr that works with Allocators. It is designed
// to be pretty much a drop-in replacement. It requires an allocator as a constructor
// argument, of course.
// Note that CssmAutoPtr<void> is perfectly valid, unlike its unique_ptr look-alike.
// You can't dereference it, naturally.
//
template <class T>
class CssmAutoPtr {
public:
	Allocator &allocator;

	CssmAutoPtr(Allocator &alloc = Allocator::standard())
	: allocator(alloc), mine(NULL) { }
	CssmAutoPtr(Allocator &alloc, T *p)
	: allocator(alloc), mine(p) { }
	CssmAutoPtr(T *p)
	: allocator(Allocator::standard()), mine(p) { }
	template <class T1> CssmAutoPtr(CssmAutoPtr<T1> &src)
	: allocator(src.allocator), mine(src.release()) { }
	template <class T1> CssmAutoPtr(Allocator &alloc, CssmAutoPtr<T1> &src)
	: allocator(alloc), mine(src.release()) { assert(allocator == src.allocator); }
	
	~CssmAutoPtr()				{ allocator.free(mine); }
	
	T *get() const _NOEXCEPT		{ return mine; }
	T *release()				{ T *result = mine; mine = NULL; return result; }
	void reset()				{ allocator.free(mine); mine = NULL; }

	operator T * () const		{ return mine; }
	T *operator -> () const		{ return mine; }
	T &operator * () const		{ assert(mine); return *mine; }

private:
	T *mine;
};

// specialization for void (i.e. void *), omitting the troublesome dereferencing ops.
template <>
class CssmAutoPtr<void> {
public:
	Allocator &allocator;

	CssmAutoPtr(Allocator &alloc) : allocator(alloc), mine(NULL) { }
	CssmAutoPtr(Allocator &alloc, void *p) : allocator(alloc), mine(p) { }
	template <class T1> CssmAutoPtr(CssmAutoPtr<T1> &src)
	: allocator(src.allocator), mine(src.release()) { }
	template <class T1> CssmAutoPtr(Allocator &alloc, CssmAutoPtr<T1> &src)
	: allocator(alloc), mine(src.release()) { assert(allocator == src.allocator); }
	
	~CssmAutoPtr()				{ destroy(mine, allocator); }
	
	void *get() _NOEXCEPT		{ return mine; }
	void *release()				{ void *result = mine; mine = NULL; return result; }
	void reset()				{ allocator.free(mine); mine = NULL; }

private:
	void *mine;
};


//
// Convenience forms of CssmAutoPtr that automatically make their (initial) object.
//
template <class T>
class CssmNewAutoPtr : public CssmAutoPtr<T> {
public:
	CssmNewAutoPtr(Allocator &alloc = Allocator::standard())
	: CssmAutoPtr<T>(alloc, new(alloc) T) { }
	
	template <class A1>
	CssmNewAutoPtr(Allocator &alloc, A1 &arg1) : CssmAutoPtr<T>(alloc, new(alloc) T(arg1)) { }
	template <class A1>
	CssmNewAutoPtr(Allocator &alloc, const A1 &arg1)
	: CssmAutoPtr<T>(alloc, new(alloc) T(arg1)) { }
	
	template <class A1, class A2>
	CssmNewAutoPtr(Allocator &alloc, A1 &arg1, A2 &arg2)
	: CssmAutoPtr<T>(alloc, new(alloc) T(arg1, arg2)) { }
	template <class A1, class A2>
	CssmNewAutoPtr(Allocator &alloc, const A1 &arg1, A2 &arg2)
	: CssmAutoPtr<T>(alloc, new(alloc) T(arg1, arg2)) { }
	template <class A1, class A2>
	CssmNewAutoPtr(Allocator &alloc, A1 &arg1, const A2 &arg2)
	: CssmAutoPtr<T>(alloc, new(alloc) T(arg1, arg2)) { }
	template <class A1, class A2>
	CssmNewAutoPtr(Allocator &alloc, const A1 &arg1, const A2 &arg2)
	: CssmAutoPtr<T>(alloc, new(alloc) T(arg1, arg2)) { }
};


} // end namespace Security


//
// Global C++ allocation hooks to use Allocators (global namespace)
//
inline void *operator new (size_t size, Allocator &allocator)
{ return allocator.malloc(size); }

inline void *operator new[] (size_t size, Allocator &allocator)
{ return allocator.malloc(size); }


#endif //_H_ALLOC
