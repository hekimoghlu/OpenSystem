/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 21, 2022.
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
//
// reqparser - interface to Requirement language parser/compiler
//
#ifndef _H_REQPARSER
#define _H_REQPARSER

#include "requirement.h"

namespace Security {
namespace CodeSigning {


//
// Generic parser interface
//
template <class ReqType>
class RequirementParser {
public:
	const ReqType *operator () (std::FILE *file);
	const ReqType *operator () (const std::string &text);
};


//
// Specifics for easier readability
//
template <class Input>
inline const Requirement *parseRequirement(const Input &source)
{ return RequirementParser<Requirement>()(source); }

template <class Input>
inline const Requirements *parseRequirements(const Input &source)
{ return RequirementParser<Requirements>()(source); }

template <class Input>
inline const BlobCore *parseGeneric(const Input &source)
{ return RequirementParser<BlobCore>()(source); }


}	// CodeSigning
}	// Security

#endif //_H_REQPARSER
