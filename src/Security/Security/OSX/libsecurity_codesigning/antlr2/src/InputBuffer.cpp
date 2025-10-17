/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 1, 2022.
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
#include "antlr/config.hpp"
#include "antlr/InputBuffer.hpp"

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
namespace antlr {
#endif

/** Ensure that the character buffer is sufficiently full */
void InputBuffer::fill(unsigned int amount)
{
	syncConsume();
	// Fill the buffer sufficiently to hold needed characters
	while (queue.entries() < amount + markerOffset)
	{
		// Append the next character
		queue.append(getChar());
	}
}

/** get the current lookahead characters as a string
 * @warning it may treat 0 and EOF values wrong
 */
ANTLR_USE_NAMESPACE(std)string InputBuffer::getLAChars( void ) const
{
	ANTLR_USE_NAMESPACE(std)string ret;

	for(unsigned int i = markerOffset; i < queue.entries(); i++)
		ret += queue.elementAt(i);

	return ret;
}

/** get the current marked characters as a string
 * @warning it may treat 0 and EOF values wrong
 */
ANTLR_USE_NAMESPACE(std)string InputBuffer::getMarkedChars( void ) const
{
	ANTLR_USE_NAMESPACE(std)string ret;

	for(unsigned int i = 0; i < markerOffset; i++)
		ret += queue.elementAt(i);

	return ret;
}

/** Return an integer marker that can be used to rewind the buffer to
 * its current state.
 */
unsigned int InputBuffer::mark()
{
	syncConsume();
	nMarkers++;
	return markerOffset;
}

/** Rewind the character buffer to a marker.
 * @param mark Marker returned previously from mark()
 */
void InputBuffer::rewind(unsigned int mark)
{
	syncConsume();
	markerOffset = mark;
	nMarkers--;
}

unsigned int InputBuffer::entries() const
{
	//assert(queue.entries() >= markerOffset);
	return (unsigned int)queue.entries() - markerOffset;
}

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
}
#endif
