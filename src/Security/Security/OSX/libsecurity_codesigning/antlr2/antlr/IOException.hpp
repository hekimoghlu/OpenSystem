/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 26, 2023.
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

#ifndef INC_IOException_hpp__
#define INC_IOException_hpp__

/* ANTLR Translator Generator
 * Project led by Terence Parr at http://www.jGuru.com
 * Software rights: http://www.antlr.org/license.html
 *
 * $Id:$
 */

#include <antlr/config.hpp>
#include <antlr/ANTLRException.hpp>
#include <exception>

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
namespace antlr {
#endif

/** Generic IOException used inside support code. (thrown by XML I/O routs)
 * basically this is something I'm using since a lot of compilers don't
 * support ios_base::failure.
 */
class ANTLR_API IOException : public ANTLRException
{
public:
	ANTLR_USE_NAMESPACE(std)exception io;

	IOException( ANTLR_USE_NAMESPACE(std)exception& e )
		: ANTLRException(e.what())
	{
	}
	IOException( const ANTLR_USE_NAMESPACE(std)string& mesg )
		: ANTLRException(mesg)
	{
	}
	virtual ~IOException() _NOEXCEPT
	{
	}
};

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
}
#endif

#endif //INC_IOException_hpp__
