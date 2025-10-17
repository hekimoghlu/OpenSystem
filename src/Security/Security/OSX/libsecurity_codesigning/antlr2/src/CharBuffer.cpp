/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 17, 2022.
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
#include "antlr/CharBuffer.hpp"
//#include <iostream>

//#include <ios>

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
namespace antlr {
#endif

/* RK: Per default istream does not throw exceptions. This can be
 * enabled with:
 * stream.exceptions(ios_base::badbit|ios_base::failbit|ios_base::eofbit);
 *
 * We could try catching the bad/fail stuff. But handling eof via this is
 * not a good idea. EOF is best handled as a 'normal' character.
 *
 * So this does not work yet with gcc... Comment it until I get to a platform
 * that does..
 */

/** Create a character buffer. Enable fail and bad exceptions, if supported
 * by platform. */
CharBuffer::CharBuffer(ANTLR_USE_NAMESPACE(std)istream& input_)
: input(input_)
{
//	input.exceptions(ANTLR_USE_NAMESPACE(std)ios_base::badbit|
//						  ANTLR_USE_NAMESPACE(std)ios_base::failbit);
}

/** Get the next character from the stream. May throw CharStreamIOException
 * when something bad happens (not EOF) (if supported by platform).
 */
int CharBuffer::getChar()
{
//	try {
    int i = input.get();
    
    if (i == -1) {
        // pass through EOF
        return -1;
    }
    
    // prevent negative-valued characters through sign extension of high-bit characters
    return static_cast<int>(static_cast<unsigned char>(i));
//	}
//	catch (ANTLR_USE_NAMESPACE(std)ios_base::failure& e) {
//		throw CharStreamIOException(e);
//	}
}

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
}
#endif
