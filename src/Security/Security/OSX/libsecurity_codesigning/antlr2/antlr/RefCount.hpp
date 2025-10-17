/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 3, 2023.
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

#ifndef INC_RefCount_hpp__
#define INC_RefCount_hpp__
/* ANTLR Translator Generator
 * Project led by Terence Parr at http://www.jGuru.com
 * Software rights: http://www.antlr.org/license.html
 *
 * $Id: //depot/code/org.antlr/release/antlr-2.7.7/lib/cpp/antlr/RefCount.hpp#2 $
 */

#include <antlr/config.hpp>

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
namespace antlr {
#endif

template<class T>
class ANTLR_API RefCount {
private:
	struct Ref {
		T* const ptr;
		unsigned int count;

		Ref(T* p) : ptr(p), count(1) {}
		~Ref() {delete ptr;}
		Ref* increment() {++count;return this;}
		bool decrement() {return (--count==0);}
	private:
		Ref(const Ref&);
		Ref& operator=(const Ref&);
	}* ref;

public:
	explicit RefCount(T* p = 0)
	: ref(p ? new Ref(p) : 0)
	{
	}
	RefCount(const RefCount<T>& other)
	: ref(other.ref ? other.ref->increment() : 0)
	{
	}
	~RefCount()
	{
		if (ref && ref->decrement())
			delete ref;
	}
	RefCount<T>& operator=(const RefCount<T>& other)
	{
		Ref* tmp = other.ref ? other.ref->increment() : 0;
		if (ref && ref->decrement())
			delete ref;
		ref = tmp;
		return *this;
	}

	operator T* () const
	{
		return ref ? ref->ptr : 0;
	}

	T* operator->() const
	{
		return ref ? ref->ptr : 0;
	}

	T* get() const
	{
		return ref ? ref->ptr : 0;
	}

	template<class newType> operator RefCount<newType>()
	{
		return RefCount<newType>(ref);
	}
};

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
}
#endif

#endif //INC_RefCount_hpp__
