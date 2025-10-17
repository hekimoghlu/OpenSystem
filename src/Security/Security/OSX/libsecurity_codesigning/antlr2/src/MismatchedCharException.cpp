/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 16, 2023.
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
#include "antlr/CharScanner.hpp"
#include "antlr/MismatchedCharException.hpp"
#include "antlr/String.hpp"

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
namespace antlr {
#endif

MismatchedCharException::MismatchedCharException()
  : RecognitionException("Mismatched char")
{}

// Expected range / not range
MismatchedCharException::MismatchedCharException(
	int c,
	int lower,
	int upper_,
	bool matchNot,
	CharScanner* scanner_
) : RecognitionException("Mismatched char",
                         scanner_->getFilename(),
								 scanner_->getLine(), scanner_->getColumn())
  , mismatchType(matchNot ? NOT_RANGE : RANGE)
  , foundChar(c)
  , expecting(lower)
  , upper(upper_)
  , scanner(scanner_)
{
}

// Expected token / not token
MismatchedCharException::MismatchedCharException(
	int c,
	int expecting_,
	bool matchNot,
	CharScanner* scanner_
) : RecognitionException("Mismatched char",
                         scanner_->getFilename(),
								 scanner_->getLine(), scanner_->getColumn())
  , mismatchType(matchNot ? NOT_CHAR : CHAR)
  , foundChar(c)
  , expecting(expecting_)
  , scanner(scanner_)
{
}

// Expected BitSet / not BitSet
MismatchedCharException::MismatchedCharException(
	int c,
	BitSet set_,
	bool matchNot,
	CharScanner* scanner_
) : RecognitionException("Mismatched char",
                         scanner_->getFilename(),
								 scanner_->getLine(), scanner_->getColumn())
  , mismatchType(matchNot ? NOT_SET : SET)
  , foundChar(c)
  , set(set_)
  , scanner(scanner_)
{
}

ANTLR_USE_NAMESPACE(std)string MismatchedCharException::getMessage() const
{
	ANTLR_USE_NAMESPACE(std)string s;

	switch (mismatchType) {
	case CHAR :
		s += "expecting '" + charName(expecting) + "', found '" + charName(foundChar) + "'";
		break;
	case NOT_CHAR :
		s += "expecting anything but '" + charName(expecting) + "'; got it anyway";
		break;
	case RANGE :
		s += "expecting token in range: '" + charName(expecting) + "'..'" + charName(upper) + "', found '" + charName(foundChar) + "'";
		break;
	case NOT_RANGE :
		s += "expecting token NOT in range: " + charName(expecting) + "'..'" + charName(upper) + "', found '" + charName(foundChar) + "'";
		break;
	case SET :
	case NOT_SET :
		{
			s += ANTLR_USE_NAMESPACE(std)string("expecting ") + (mismatchType == NOT_SET ? "NOT " : "") + "one of (";
			ANTLR_USE_NAMESPACE(std)vector<unsigned int> elems = set.toArray();
			for ( unsigned int i = 0; i < elems.size(); i++ )
			{
				s += " '";
				s += charName(elems[i]);
				s += "'";
			}
			s += "), found '" + charName(foundChar) + "'";
		}
		break;
	default :
		s += RecognitionException::getMessage();
		break;
	}

	return s;
}

#ifndef NO_STATIC_CONSTS
const int MismatchedCharException::CHAR;
const int MismatchedCharException::NOT_CHAR;
const int MismatchedCharException::RANGE;
const int MismatchedCharException::NOT_RANGE;
const int MismatchedCharException::SET;
const int MismatchedCharException::NOT_SET;
#endif

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
}
#endif
