/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 22, 2025.
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
 * cssm utilities
 */
#ifndef _H_ENDIAN
#define _H_ENDIAN

#include <machine/endian.h>
#include <libkern/OSByteOrder.h>
#include <security_utilities/utilities.h>
#include <security_utilities/memutils.h>
#include <security_utilities/debugging.h>

namespace Security {


//
// Encode/decode operations by type, overloaded.
// You can use these functions directly, but consider using
// the higher-level constructs below instead.
//
#ifdef __LP64__
static inline unsigned long h2n(unsigned long v) { return OSSwapHostToBigInt64(v); }
static inline unsigned long n2h(unsigned long v) { return OSSwapBigToHostInt64(v); }
static inline unsigned long flip(unsigned long v) { return OSSwapInt64(v); }
static inline signed long h2n(signed long v) { return OSSwapHostToBigInt64(v); }
static inline signed long n2h(signed long v) { return OSSwapBigToHostInt64(v); }
static inline signed long flip(signed long v) { return OSSwapInt64(v); }
#else
static inline unsigned long h2n(unsigned long v)	{ return htonl(v); }
static inline unsigned long n2h(unsigned long v)	{ return ntohl(v); }
static inline unsigned long flip(unsigned long v)	{ return OSSwapInt32(v); }
static inline signed long h2n(signed long v)		{ return htonl(v); }
static inline signed long n2h(signed long v)		{ return ntohl(v); }
static inline signed long flip(signed long v)		{ return OSSwapInt32(v); }
#endif

static inline unsigned long long h2n(unsigned long long v) { return OSSwapHostToBigInt64(v); }
static inline unsigned long long n2h(unsigned long long v) { return OSSwapBigToHostInt64(v); }
static inline unsigned long long flip(unsigned long long v) { return OSSwapInt64(v); }
static inline long long h2n(long long v)			{ return OSSwapHostToBigInt64(v); }
static inline long long n2h(long long v)			{ return OSSwapBigToHostInt64(v); }
static inline long long flip(long long v)			{ return OSSwapInt64(v); }

static inline unsigned int h2n(unsigned int v)		{ return htonl(v); }
static inline unsigned int n2h(unsigned int v)		{ return ntohl(v); }
static inline unsigned int flip(unsigned int v)		{ return OSSwapInt32(v); }
static inline signed int h2n(int v)					{ return htonl(v); }
static inline signed int n2h(int v)					{ return ntohl(v); }
static inline signed int flip(int v)				{ return OSSwapInt32(v); }

static inline unsigned short h2n(unsigned short v)	{ return htons(v); }
static inline unsigned short n2h(unsigned short v)	{ return ntohs(v); }
static inline unsigned short flip(unsigned short v)	{ return OSSwapInt16(v); }
static inline signed short h2n(signed short v)		{ return htons(v); }
static inline signed short n2h(signed short v)		{ return ntohs(v); }
static inline signed short flip(signed short v)		{ return OSSwapInt16(v); }

static inline unsigned char h2n(unsigned char v)	{ return v; }
static inline unsigned char n2h(unsigned char v)	{ return v; }
static inline unsigned char flip(unsigned char v)	{ return v; }
static inline signed char h2n(signed char v)		{ return v; }
static inline signed char n2h(signed char v)		{ return v; }
static inline signed char flip(signed char v)		{ return v; }


//
// Flip pointers
//
template <class Base>
static inline Base *h2n(Base *p)	{ return (Base *)h2n(uintptr_t(p)); }

template <class Base>
static inline Base *n2h(Base *p)	{ return (Base *)n2h(uintptr_t(p)); }


//
// In-place fix operations
//
template <class Type>
static inline void h2ni(Type &v)	{ v = h2n(v); }

template <class Type>
static inline void n2hi(Type &v)	{ v = n2h(v); }

//
// Endian<SomeType> keeps NBO values in memory and converts
// during loads and stores. This presumes that you are using
// memory blocks thare are read/written/mapped as amorphous byte
// streams, but want to be byte-order clean using them.
//
// The generic definition uses h2n/n2h to flip bytes. Feel free
// to declare specializations of Endian<T> as appropriate.
//
// Note well that the address of an Endian<T> is not an address-of-T,
// and there is no conversion available.
//
template <class Type>
class Endian {
public:
    typedef Type Value;
    Endian() : mValue(Type(0)) { }
    Endian(Value v) : mValue((Type) h2n(v)) { }
    
    Type get ()	const			{ return (Type) n2h(mValue); }
    operator Value () const		{ return this->get(); }
    Endian &operator = (Value v)	{ mValue = h2n(v); return *this; }

    
private:
    Value mValue;
} __attribute__((packed));


}	// end namespace Security


#endif //_H_ENDIAN
