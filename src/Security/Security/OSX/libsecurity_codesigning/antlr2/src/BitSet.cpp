/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 11, 2023.
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
#include "antlr/BitSet.hpp"
#include <string>

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
namespace antlr {
#endif

BitSet::BitSet(unsigned int nbits)
: storage(nbits)
{
	for (unsigned int i = 0; i < nbits ; i++ )
		storage[i] = false;
}

BitSet::BitSet( const unsigned long* bits_, unsigned int nlongs )
: storage(nlongs*32)
{
	for ( unsigned int i = 0 ; i < (nlongs * 32); i++)
		storage[i] = (bits_[i>>5] & (1UL << (i&31))) ? true : false;
}

BitSet::~BitSet()
{
}

void BitSet::add(unsigned int el)
{
	if( el >= storage.size() )
		storage.resize( el+1, false );

	storage[el] = true;
}

bool BitSet::member(unsigned int el) const
{
	if ( el >= storage.size())
		return false;

	return storage[el];
}

ANTLR_USE_NAMESPACE(std)vector<unsigned int> BitSet::toArray() const
{
	ANTLR_USE_NAMESPACE(std)vector<unsigned int> elems;
	for (unsigned int i = 0; i < storage.size(); i++)
	{
		if (storage[i])
			elems.push_back(i);
	}

	return elems;
}

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
}
#endif
