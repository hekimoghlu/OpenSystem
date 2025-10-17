/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 8, 2025.
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

//===- unittest/Support/YAMLParserTest ------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/YAMLParser.h"
#include "gtest/gtest.h"

namespace llvm {

static void SuppressDiagnosticsOutput(const SMDiagnostic &, void *) {
  // Prevent SourceMgr from writing errors to stderr 
  // to reduce noise in unit test runs.
}

// Checks that the given input gives a parse error. Makes sure that an error
// text is available and the parse fails.
static void ExpectParseError(StringRef Message, StringRef Input) {
  SourceMgr SM;
  yaml::Stream Stream(Input, SM);
  SM.setDiagHandler(SuppressDiagnosticsOutput);
  EXPECT_FALSE(Stream.validate()) << Message << ": " << Input;
  EXPECT_TRUE(Stream.failed()) << Message << ": " << Input;
}

// Checks that the given input can be parsed without error.
static void ExpectParseSuccess(StringRef Message, StringRef Input) {
  SourceMgr SM;
  yaml::Stream Stream(Input, SM);
  EXPECT_TRUE(Stream.validate()) << Message << ": " << Input;
}

TEST(YAMLParser, ParsesEmptyArray) {
  ExpectParseSuccess("Empty array", "[]");
}

TEST(YAMLParser, FailsIfNotClosingArray) {
  ExpectParseError("Not closing array", "[");
  ExpectParseError("Not closing array", "  [  ");
  ExpectParseError("Not closing array", "  [x");
}

TEST(YAMLParser, ParsesEmptyArrayWithWhitespace) {
  ExpectParseSuccess("Array with spaces", "  [  ]  ");
  ExpectParseSuccess("All whitespaces", "\t\r\n[\t\n \t\r ]\t\r \n\n");
}

TEST(YAMLParser, ParsesEmptyObject) {
  ExpectParseSuccess("Empty object", "[{}]");
}

TEST(YAMLParser, ParsesObject) {
  ExpectParseSuccess("Object with an entry", "[{\"a\":\"/b\"}]");
}

TEST(YAMLParser, ParsesMultipleKeyValuePairsInObject) {
  ExpectParseSuccess("Multiple key, value pairs",
                     "[{\"a\":\"/b\",\"c\":\"d\",\"e\":\"f\"}]");
}

TEST(YAMLParser, FailsIfNotClosingObject) {
  ExpectParseError("Missing close on empty", "[{]");
  ExpectParseError("Missing close after pair", "[{\"a\":\"b\"]");
}

TEST(YAMLParser, FailsIfMissingColon) {
  ExpectParseError("Missing colon between key and value", "[{\"a\"\"/b\"}]");
  ExpectParseError("Missing colon between key and value", "[{\"a\" \"b\"}]");
}

TEST(YAMLParser, FailsOnMissingQuote) {
  ExpectParseError("Missing open quote", "[{a\":\"b\"}]");
  ExpectParseError("Missing closing quote", "[{\"a\":\"b}]");
}

TEST(YAMLParser, ParsesEscapedQuotes) {
  ExpectParseSuccess("Parses escaped string in key and value",
                     "[{\"a\":\"\\\"b\\\"  \\\" \\\"\"}]");
}

TEST(YAMLParser, ParsesEmptyString) {
  ExpectParseSuccess("Parses empty string in value", "[{\"a\":\"\"}]");
}

TEST(YAMLParser, ParsesMultipleObjects) {
  ExpectParseSuccess(
      "Multiple objects in array",
      "["
      " { \"a\" : \"b\" },"
      " { \"a\" : \"b\" },"
      " { \"a\" : \"b\" }"
      "]");
}

TEST(YAMLParser, FailsOnMissingComma) {
  ExpectParseError(
      "Missing comma",
      "["
      " { \"a\" : \"b\" }"
      " { \"a\" : \"b\" }"
      "]");
}

TEST(YAMLParser, ParsesSpacesInBetweenTokens) {
  ExpectParseSuccess(
      "Various whitespace between tokens",
      " \t \n\n \r [ \t \n\n \r"
      " \t \n\n \r { \t \n\n \r\"a\"\t \n\n \r :"
      " \t \n\n \r \"b\"\t \n\n \r } \t \n\n \r,\t \n\n \r"
      " \t \n\n \r { \t \n\n \r\"a\"\t \n\n \r :"
      " \t \n\n \r \"b\"\t \n\n \r } \t \n\n \r]\t \n\n \r");
}

TEST(YAMLParser, ParsesArrayOfArrays) {
  ExpectParseSuccess("Array of arrays", "[[]]");
}

TEST(YAMLParser, HandlesEndOfFileGracefully) {
  ExpectParseError("In string starting with EOF", "[\"");
  ExpectParseError("In string hitting EOF", "[\"   ");
  ExpectParseError("In string escaping EOF", "[\"  \\");
  ExpectParseError("In array starting with EOF", "[");
  ExpectParseError("In array element starting with EOF", "[[], ");
  ExpectParseError("In array hitting EOF", "[[] ");
  ExpectParseError("In array hitting EOF", "[[]");
  ExpectParseError("In object hitting EOF", "{\"\"");
}

// Checks that the given string can be parsed into an identical string inside
// of an array.
static void ExpectCanParseString(StringRef String) {
  std::string StringInArray = (llvm::Twine("[\"") + String + "\"]").str();
  SourceMgr SM;
  yaml::Stream Stream(StringInArray, SM);
  yaml::SequenceNode *ParsedSequence
    = dyn_cast<yaml::SequenceNode>(Stream.begin()->getRoot());
  StringRef ParsedString
    = dyn_cast<yaml::ScalarNode>(
      static_cast<yaml::Node*>(ParsedSequence->begin()))->getRawValue();
  ParsedString = ParsedString.substr(1, ParsedString.size() - 2);
  EXPECT_EQ(String, ParsedString.str());
}

// Checks that parsing the given string inside an array fails.
static void ExpectCannotParseString(StringRef String) {
  std::string StringInArray = (llvm::Twine("[\"") + String + "\"]").str();
  ExpectParseError((Twine("When parsing string \"") + String + "\"").str(),
                   StringInArray);
}

TEST(YAMLParser, ParsesStrings) {
  ExpectCanParseString("");
  ExpectCannotParseString("\\");
  ExpectCannotParseString("\"");
  ExpectCanParseString(" ");
  ExpectCanParseString("\\ ");
  ExpectCanParseString("\\\"");
  ExpectCannotParseString("\"\\");
  ExpectCannotParseString(" \\");
  ExpectCanParseString("\\\\");
  ExpectCannotParseString("\\\\\\");
  ExpectCanParseString("\\\\\\\\");
  ExpectCanParseString("\\\" ");
  ExpectCannotParseString("\\\\\" ");
  ExpectCanParseString("\\\\\\\" ");
  ExpectCanParseString("    \\\\  \\\"  \\\\\\\"   ");
}

TEST(YAMLParser, WorksWithIteratorAlgorithms) {
  SourceMgr SM;
  yaml::Stream Stream("[\"1\", \"2\", \"3\", \"4\", \"5\", \"6\"]", SM);
  yaml::SequenceNode *Array
    = dyn_cast<yaml::SequenceNode>(Stream.begin()->getRoot());
  EXPECT_EQ(6, std::distance(Array->begin(), Array->end()));
}

} // end namespace llvm
