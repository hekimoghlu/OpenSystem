/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 27, 2024.
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
// reqmaker - Requirement assembler
//
#ifndef _H_REQMAKER
#define _H_REQMAKER

#include "requirement.h"

namespace Security {
namespace CodeSigning {


//
// A Requirement::Maker is a tool for creating a Requirement blob.
// It's primarily an assember for the binary requirements (exprOp) language.
// Initialize it, call put() methods to generate the exprOp program, then
// call make() to get the assembled Requirement blob, malloc'ed for you.
// The Maker is not reusable.
//
class Requirement::Maker {
public:
	Maker(Kind k = exprForm);
	~Maker() { free(mBuffer); }
	
	template <class T>
	T *alloc(size_t size) { return reinterpret_cast<T *>(alloc(size)); }

	template <class T>
	void put(const T &value) { *alloc<Endian<T> >(sizeof(T)) = value; }
	void put(ExprOp op) { put(uint32_t(op)); }
	void put(MatchOperation op) { put(uint32_t(op)); }
	void put(const std::string &s) { putData(s.data(), s.size()); }
	void put(const char *s) { putData(s, strlen(s)); }
	void putData(const void *data, size_t length);
	void putData(CFStringRef s) { put(cfString(s)); }
	
	void anchor(int slot, SHA1::Digest digest);			// given slot/digest
	void anchor(int slot, const void *cert, size_t length); // given slot/cert
	void anchor();										// made-by-Apple
	void anchorGeneric();								// anything drawn from the Apple anchor
	
	void trustedAnchor();
	void trustedAnchor(int slot);
	
	void infoKey(const std::string &key, const std::string &value);
	void ident(const std::string &identHash);
	void cdhash(SHA1::Digest digest);
	void cdhash(CFDataRef digest);
	void platform(int platformIdentifier);

	void copy(const void *data, size_t length)
		{ memcpy(this->alloc(length), data, length); }
	void copy(const Requirement *req);				// inline expand
	
	//
	// Keep labels into exprOp code, and allow for "shifting in"
	// prefix code as needed (exprOp is a prefix-code language).
	//
	struct Label {
		const Offset pos;
		Label(const Maker &maker) : pos((const Offset)maker.length()) { }
	};
	void *insert(const Label &label, size_t length = sizeof(uint32_t));
	
	template <class T>
	Endian<T> &insert(const Label &label, size_t length = sizeof(T))
	{ return *reinterpret_cast<Endian<T>*>(insert(label, length)); }

	//
	// Help with making operator chains (foo AND bar AND baz...).
	// Note that the empty case (no elements at all) must be resolved by the caller.
	//
	class Chain : public Label {
	public:
		Chain(Maker &myMaker, ExprOp op)
			: Label(myMaker), maker(myMaker), mJoiner(op), mCount(0) { }

		void add() const
			{ if (mCount++) maker.insert<ExprOp>(*this) = mJoiner; }
	
		Maker &maker;
		bool empty() const { return mCount == 0; }

	private:
		ExprOp mJoiner;
		mutable unsigned mCount;
	};
	
	
	//
	// Over-all construction management
	//
	void kind(Kind k) { mBuffer->kind(k); }
	size_t length() const { return mPC; }
	Requirement *make();
	Requirement *operator () () { return make(); }
	
protected:
	void require(size_t size);	
	void *alloc(size_t size);

private:
	Requirement *mBuffer;
	Offset mSize;
	Offset mPC;
};


}	// CodeSigning
}	// Security

#endif //_H_REQMAKER
