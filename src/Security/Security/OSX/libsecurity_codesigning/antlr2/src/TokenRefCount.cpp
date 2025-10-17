/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 20, 2022.
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
#include "antlr/TokenRefCount.hpp"
#include "antlr/Token.hpp"

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
namespace antlr {
#endif

TokenRef::TokenRef(Token* p)
: ptr(p), count(1)
{
	if (p && !p->ref)
		p->ref = this;
}

TokenRef::~TokenRef()
{
	delete ptr;
}

TokenRef* TokenRef::getRef(const Token* p)
{
	if (p) {
		Token* pp = const_cast<Token*>(p);
		if (pp->ref)
			return pp->ref->increment();
		else
			return new TokenRef(pp);
	} else
		return 0;
}

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
}
#endif

