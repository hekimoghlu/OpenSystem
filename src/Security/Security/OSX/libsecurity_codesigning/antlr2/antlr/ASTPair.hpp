/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 22, 2023.
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

#ifndef INC_ASTPair_hpp__
#define INC_ASTPair_hpp__

/* ANTLR Translator Generator
 * Project led by Terence Parr at http://www.jGuru.com
 * Software rights: http://www.antlr.org/license.html
 *
 * $Id: //depot/code/org.antlr/release/antlr-2.7.7/lib/cpp/antlr/ASTPair.hpp#2 $
 */

#include <antlr/config.hpp>
#include <antlr/AST.hpp>

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
namespace antlr {
#endif

/** ASTPair:  utility class used for manipulating a pair of ASTs
  * representing the current AST root and current AST sibling.
  * This exists to compensate for the lack of pointers or 'var'
  * arguments in Java.
  *
  * OK, so we can do those things in C++, but it seems easier
  * to stick with the Java way for now.
  */
class ANTLR_API ASTPair {
public:
	RefAST root;		// current root of tree
	RefAST child;		// current child to which siblings are added

	/** Make sure that child is the last sibling */
	void advanceChildToEnd() {
		if (child) {
			while (child->getNextSibling()) {
				child = child->getNextSibling();
			}
		}
	}
//	/** Copy an ASTPair.  Don't call it clone() because we want type-safety */
//	ASTPair copy() {
//		ASTPair tmp = new ASTPair();
//		tmp.root = root;
//		tmp.child = child;
//		return tmp;
//	}
	ANTLR_USE_NAMESPACE(std)string toString() const {
		ANTLR_USE_NAMESPACE(std)string r = !root ? ANTLR_USE_NAMESPACE(std)string("null") : root->getText();
		ANTLR_USE_NAMESPACE(std)string c = !child ? ANTLR_USE_NAMESPACE(std)string("null") : child->getText();
		return "["+r+","+c+"]";
	}
};

#ifdef ANTLR_CXX_SUPPORTS_NAMESPACE
}
#endif

#endif //INC_ASTPair_hpp__
