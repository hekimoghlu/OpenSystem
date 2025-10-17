/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 1, 2024.
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

#ifndef INC_TokenStreamRecognitionException_hpp__
#define INC_TokenStreamRecognitionException_hpp__

/* ANTLR Translator Generator
 * Project led by Terence Parr at http://www.jGuru.com
 * Software rights: http://www.antlr.org/license.html
 *
 * $Id: //depot/code/org.antlr/release/antlr-2.7.7/lib/cpp/antlr/TokenStreamRecognitionException.hpp#2 $
 */

#include <antlr/config.hpp>
#include <antlr/TokenStreamException.hpp>

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
namespace antlr {
#endif

/** Exception thrown from generated lexers when there's no default error
 * handler specified.
 * @see TokenStream
 */
class TokenStreamRecognitionException : public TokenStreamException {
public:
	TokenStreamRecognitionException(RecognitionException& re)
	: TokenStreamException(re.getMessage())
	, recog(re)
	{
	}
	virtual ~TokenStreamRecognitionException() _NOEXCEPT
	{
	}
	virtual ANTLR_USE_NAMESPACE(std)string toString() const
	{
		return recog.getFileLineColumnString()+getMessage();
	}

	virtual ANTLR_USE_NAMESPACE(std)string getFilename() const _NOEXCEPT
	{
		return recog.getFilename();
	}
	virtual int getLine() const _NOEXCEPT
	{
		return recog.getLine();
	}
	virtual int getColumn() const _NOEXCEPT
	{
		return recog.getColumn();
	}
private:
	RecognitionException recog;
};

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
}
#endif

#endif //INC_TokenStreamRecognitionException_hpp__
