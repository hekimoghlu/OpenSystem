/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 4, 2024.
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

#ifndef INC_CommonHiddenStreamToken_hpp__
#define INC_CommonHiddenStreamToken_hpp__

/* ANTLR Translator Generator
 * Project led by Terence Parr at http://www.jGuru.com
 * Software rights: http://www.antlr.org/license.html
 *
 * $Id: //depot/code/org.antlr/release/antlr-2.7.7/lib/cpp/antlr/CommonHiddenStreamToken.hpp#2 $
 */

#include <antlr/config.hpp>
#include <antlr/CommonToken.hpp>

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
namespace antlr {
#endif

class ANTLR_API CommonHiddenStreamToken : public CommonToken {
protected:
	RefToken hiddenBefore;
	RefToken hiddenAfter;

public:
	CommonHiddenStreamToken();
	CommonHiddenStreamToken(int t, const ANTLR_USE_NAMESPACE(std)string& txt);
	CommonHiddenStreamToken(const ANTLR_USE_NAMESPACE(std)string& s);

	RefToken getHiddenAfter();
	RefToken getHiddenBefore();

   static RefToken factory();

	void setHiddenAfter(RefToken t);
	void setHiddenBefore(RefToken t);
};

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
}
#endif

#endif //INC_CommonHiddenStreamToken_hpp__
