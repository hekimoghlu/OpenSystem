/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 3, 2022.
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
#include "antlrplugin.h"
#include "cserror.h"
#include "RequirementLexer.hpp"
#include "RequirementParser.hpp"
#include <antlr/TokenStreamException.hpp>


namespace Security {
namespace CodeSigning {

namespace Parser = Security_CodeSigning;


//
// Lexer input adapters
//
class StdioInputStream : public antlr::InputBuffer {
public:
	StdioInputStream(FILE *fp) : mFile(fp) { }	
	int getChar() { return fgetc(mFile); }

private:
	FILE *mFile;
};

class StringInputStream : public antlr::InputBuffer {
public:
	StringInputStream(const string &s) : mInput(s), mPos(mInput.begin()) { }
	int getChar() { return (mPos == mInput.end()) ? EOF : static_cast<unsigned char>(*mPos++); }

private:
	string mInput;
	string::const_iterator mPos;
};


//
// Generic parser driver
//
template <class Input, class Source, class Result>
const Result *parse(Source source, Result *(Parser::RequirementParser::*rule)(), std::string &errors)
{
	Input input(source);
	Parser::RequirementLexer lexer(input);
	Parser::RequirementParser parser(lexer);
	try {
		const Result *result = (parser.*rule)();
		errors = parser.errors;
		if (errors.empty())
			return result;
		else
			::free((void *)result);
	} catch (const antlr::TokenStreamException &ex) {
		errors = ex.toString() + "\n";
	}
	return NULL;			// signal failure
}


//
// Hook up each supported parsing action to the plugin interface
//
static
const Requirement *fileRequirement(FILE *source, string &errors)
{ return parse<StdioInputStream>(source, &Parser::RequirementParser::requirement, errors); }

static
const Requirement *stringRequirement(string source, string &errors)
{ return parse<StringInputStream>(source, &Parser::RequirementParser::requirement, errors); }

static
const Requirements *fileRequirements(FILE *source, string &errors)
{ return parse<StdioInputStream>(source, &Parser::RequirementParser::requirementSet, errors); }

static
const Requirements *stringRequirements(string source, string &errors)
{ return parse<StringInputStream>(source, &Parser::RequirementParser::requirementSet, errors); }

static
const BlobCore *fileGeneric(FILE *source, string &errors)
{ return parse<StdioInputStream>(source, &Parser::RequirementParser::autosense, errors); }

static
const BlobCore *stringGeneric(string source, string &errors)
{ return parse<StringInputStream>(source, &Parser::RequirementParser::autosense, errors); }


//
// Basic plugin hookup
//
static AntlrPlugin plugin = {
	fileRequirement,
	fileRequirements,
	fileGeneric,
	stringRequirement,
	stringRequirements,
	stringGeneric
};

AntlrPlugin *findAntlrPlugin()
{
	return &plugin;
}


} // end namespace CodeSigning
} // end namespace Security
