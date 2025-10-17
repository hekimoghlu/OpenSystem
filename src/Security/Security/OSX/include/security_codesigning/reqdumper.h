/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 26, 2023.
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
// reqdumper - Requirement un-parsing (disassembly)
//
#ifndef _H_REQDUMPER
#define _H_REQDUMPER

#include "reqreader.h"
#include <ctype.h>


namespace Security {
namespace CodeSigning {


//
// A decompiler for (compiled) requirements programs.
// This is intended to produce compiler-ready source, and the
// (decompile . compile) cycle is meant to be loss-less.
//
// Note that a Dumper is a type of Interpreter, so it can use the program stream
// accessors of the Interpreter. However, the evaluaton Context is absent, so
// actual validation functions must not be called.
//
class Dumper : public Requirement::Reader {
public:
	explicit Dumper(const Requirement *req, bool debug = false)
		: Reader(req), mDebug(debug) { }
	
	enum SyntaxLevel {
		slPrimary,		// syntax primary
		slAnd,			// conjunctive
		slOr,			// disjunctive
		slTop			// where we start
	};
	
	void dump();		// decompile this (entire) requirement
	void expr(SyntaxLevel level = slTop); // decompile one requirement expression
	
	std::string value() const { return mOutput; }
	operator std::string () const { return value(); }
	
	typedef unsigned char Byte;
	
public:
	// all-in-one dumping
	static string dump(const Requirements *reqs, bool debug = false);
	static string dump(const Requirement *req, bool debug = false);
	static string dump(const BlobCore *req, bool debug = false);	// dumps either

protected:
	enum PrintMode {
		isSimple,		// printable and does not require quotes
		isPrintable,	// can be quoted safely
		isBinary		// contains binary bytes (use 0xnnn form)
	};
	void data(PrintMode bestMode = isSimple, bool dotOkay = false);
	void timestamp();
	void dotString() { data(isSimple, true); }
	void quotedString() { data(isPrintable); }
	void hashData();	// H"bytes"
	void certSlot();	// symbolic certificate slot indicator (explicit)
	void match();		// a match suffix (op + value)
	
	void print(const char *format, ...) __attribute((format(printf,2,3)));

private:
	void printBytes(const Byte *data, size_t length); // just write hex bytes
	
private:
	std::string mOutput;		// output accumulator
	bool mDebug;				// include debug output in mOutput
};


}	// CodeSigning
}	// Security

#endif //_H_REQDUMPER
