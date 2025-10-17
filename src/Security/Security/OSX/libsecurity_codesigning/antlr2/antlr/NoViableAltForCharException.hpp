/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 11, 2024.
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

#ifndef INC_NoViableAltForCharException_hpp__
# define INC_NoViableAltForCharException_hpp__

/* ANTLR Translator Generator
 * Project led by Terence Parr at http://www.jGuru.com
 * Software rights: http://www.antlr.org/license.html
 *
 * $Id: //depot/code/org.antlr/release/antlr-2.7.7/lib/cpp/antlr/NoViableAltForCharException.hpp#2 $
 */

# include <antlr/config.hpp>
# include <antlr/RecognitionException.hpp>
# include <antlr/CharScanner.hpp>

# ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
namespace antlr
{
# endif

class ANTLR_API NoViableAltForCharException : public RecognitionException
{
public:
	NoViableAltForCharException(int c, CharScanner* scanner);
	NoViableAltForCharException(int c, const ANTLR_USE_NAMESPACE(std)string& fileName_,
										 int line_, int column_);

	virtual ~NoViableAltForCharException() _NOEXCEPT
	{
	}

	/// Returns a clean error message (no line number/column information)
	ANTLR_USE_NAMESPACE(std)string getMessage() const;
protected:
	int foundChar;
};

# ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
}
# endif

#endif //INC_NoViableAltForCharException_hpp__
