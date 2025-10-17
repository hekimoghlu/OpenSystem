/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 28, 2023.
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

#ifndef INC_RecognitionException_hpp__
# define INC_RecognitionException_hpp__

/* ANTLR Translator Generator
 * Project led by Terence Parr at http://www.jGuru.com
 * Software rights: http://www.antlr.org/license.html
 *
 * $Id: //depot/code/org.antlr/release/antlr-2.7.7/lib/cpp/antlr/RecognitionException.hpp#2 $
 */

# include <antlr/config.hpp>
# include <antlr/ANTLRException.hpp>

# ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
namespace antlr
{
# endif
	class ANTLR_API RecognitionException : public ANTLRException
	{
	public:
		RecognitionException();
		RecognitionException(const ANTLR_USE_NAMESPACE(std)string& s);
		RecognitionException(const ANTLR_USE_NAMESPACE(std)string& s,
									const ANTLR_USE_NAMESPACE(std)string& fileName,
									int line, int column );

		virtual ~RecognitionException() _NOEXCEPT
		{
		}

		/// Return file where mishap occurred.
		virtual ANTLR_USE_NAMESPACE(std)string getFilename() const _NOEXCEPT
		{
			return fileName;
		}
		/**
		 * @return the line number that this exception happened on.
		 */
		virtual int getLine() const _NOEXCEPT
		{
			return line;
		}
		/**
		 * @return the column number that this exception happened on.
		 */
		virtual int getColumn() const _NOEXCEPT
		{
			return column;
		}

		/// Return complete error message with line/column number info (if present)
		virtual ANTLR_USE_NAMESPACE(std)string toString() const;

		/// See what file/line/column info is present and return it as a string
		virtual ANTLR_USE_NAMESPACE(std)string getFileLineColumnString() const;
	protected:
		ANTLR_USE_NAMESPACE(std)string fileName; // not used by treeparsers
		int line;    // not used by treeparsers
		int column;  // not used by treeparsers
	};

# ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
}
# endif

#endif //INC_RecognitionException_hpp__
