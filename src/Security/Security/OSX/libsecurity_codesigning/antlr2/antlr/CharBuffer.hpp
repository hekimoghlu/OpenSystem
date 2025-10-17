/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 12, 2023.
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

#ifndef INC_CharBuffer_hpp__
#define INC_CharBuffer_hpp__

/* ANTLR Translator Generator
 * Project led by Terence Parr at http://www.jGuru.com
 * Software rights: http://www.antlr.org/license.html
 *
 * $Id: //depot/code/org.antlr/release/antlr-2.7.7/lib/cpp/antlr/CharBuffer.hpp#2 $
 */

#include <antlr/config.hpp>

#include <istream>

#include <antlr/InputBuffer.hpp>

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
namespace antlr {
#endif

/**A Stream of characters fed to the lexer from a InputStream that can
 * be rewound via mark()/rewind() methods.
 * <p>
 * A dynamic array is used to buffer up all the input characters.  Normally,
 * "k" characters are stored in the buffer.  More characters may be stored
 * during guess mode (testing syntactic predicate), or when LT(i>k) is
 * referenced.
 * Consumption of characters is deferred.  In other words, reading the next
 * character is not done by consume(), but deferred until needed by LA or LT.
 * <p>
 *
 * @see antlr.CharQueue
 */

class ANTLR_API CharBuffer : public InputBuffer {
public:
	/// Create a character buffer
	CharBuffer( ANTLR_USE_NAMESPACE(std)istream& input );
	/// Get the next character from the stream
	int getChar();

protected:
	// character source
	ANTLR_USE_NAMESPACE(std)istream& input;

private:
	// NOTE: Unimplemented
	CharBuffer(const CharBuffer& other);
	CharBuffer& operator=(const CharBuffer& other);
};

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
}
#endif

#endif //INC_CharBuffer_hpp__
