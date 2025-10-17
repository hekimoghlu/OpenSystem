/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 14, 2025.
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
#include "antlr/RecognitionException.hpp"
#include "antlr/String.hpp"

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
namespace antlr {
#endif

RecognitionException::RecognitionException()
: ANTLRException("parsing error")
, line(-1)
, column(-1)
{
}

RecognitionException::RecognitionException(const ANTLR_USE_NAMESPACE(std)string& s)
: ANTLRException(s)
, line(-1)
, column(-1)
{
}

RecognitionException::RecognitionException(const ANTLR_USE_NAMESPACE(std)string& s,
                                           const ANTLR_USE_NAMESPACE(std)string& fileName_,
                                           int line_,int column_)
: ANTLRException(s)
, fileName(fileName_)
, line(line_)
, column(column_)
{
}

ANTLR_USE_NAMESPACE(std)string RecognitionException::getFileLineColumnString() const
{
	ANTLR_USE_NAMESPACE(std)string fileLineColumnString;

	if ( fileName.length() > 0 )
		fileLineColumnString = fileName + ":";

	if ( line != -1 )
	{
		if ( fileName.length() == 0 )
			fileLineColumnString = fileLineColumnString + "line ";

		fileLineColumnString = fileLineColumnString + line;

		if ( column != -1 )
			fileLineColumnString = fileLineColumnString + ":" + column;

		fileLineColumnString = fileLineColumnString + ":";
	}

	fileLineColumnString = fileLineColumnString + " ";

	return fileLineColumnString;
}

ANTLR_USE_NAMESPACE(std)string RecognitionException::toString() const
{
	return getFileLineColumnString()+getMessage();
}

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
}
#endif
