/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 2, 2025.
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

//===- llvm/unittest/Support/raw_ostream_test.cpp - raw_ostream tests -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

template<typename T> std::string printToString(const T &Value) {
  std::string res;
  llvm::raw_string_ostream(res) << Value;
  return res;    
}

/// printToString - Print the given value to a stream which only has \arg
/// BytesLeftInBuffer bytes left in the buffer. This is useful for testing edge
/// cases in the buffer handling logic.
template<typename T> std::string printToString(const T &Value,
                                               unsigned BytesLeftInBuffer) {
  // FIXME: This is relying on internal knowledge of how raw_ostream works to
  // get the buffer position right.
  SmallString<256> SVec;
  assert(BytesLeftInBuffer < 256 && "Invalid buffer count!");
  llvm::raw_svector_ostream OS(SVec);
  unsigned StartIndex = 256 - BytesLeftInBuffer;
  for (unsigned i = 0; i != StartIndex; ++i)
    OS << '?';
  OS << Value;
  return OS.str().substr(StartIndex);
}

template<typename T> std::string printToStringUnbuffered(const T &Value) {
  std::string res;
  llvm::raw_string_ostream OS(res);
  OS.SetUnbuffered();
  OS << Value;
  return res;
}

TEST(raw_ostreamTest, Types_Buffered) {
  // Char
  EXPECT_EQ("c", printToString('c'));

  // String
  EXPECT_EQ("hello", printToString("hello"));
  EXPECT_EQ("hello", printToString(std::string("hello")));

  // Int
  EXPECT_EQ("0", printToString(0));
  EXPECT_EQ("2425", printToString(2425));
  EXPECT_EQ("-2425", printToString(-2425));

  // Long long
  EXPECT_EQ("0", printToString(0LL));
  EXPECT_EQ("257257257235709", printToString(257257257235709LL));
  EXPECT_EQ("-257257257235709", printToString(-257257257235709LL));

  // Double
  EXPECT_EQ("1.100000e+00", printToString(1.1));

  // void*
  EXPECT_EQ("0x0", printToString((void*) 0));
  EXPECT_EQ("0xbeef", printToString((void*) 0xbeef));
  EXPECT_EQ("0xdeadbeef", printToString((void*) 0xdeadbeef));

  // Min and max.
  EXPECT_EQ("18446744073709551615", printToString(UINT64_MAX));
  EXPECT_EQ("-9223372036854775808", printToString(INT64_MIN));
}

TEST(raw_ostreamTest, Types_Unbuffered) {  
  // Char
  EXPECT_EQ("c", printToStringUnbuffered('c'));

  // String
  EXPECT_EQ("hello", printToStringUnbuffered("hello"));
  EXPECT_EQ("hello", printToStringUnbuffered(std::string("hello")));

  // Int
  EXPECT_EQ("0", printToStringUnbuffered(0));
  EXPECT_EQ("2425", printToStringUnbuffered(2425));
  EXPECT_EQ("-2425", printToStringUnbuffered(-2425));

  // Long long
  EXPECT_EQ("0", printToStringUnbuffered(0LL));
  EXPECT_EQ("257257257235709", printToStringUnbuffered(257257257235709LL));
  EXPECT_EQ("-257257257235709", printToStringUnbuffered(-257257257235709LL));

  // Double
  EXPECT_EQ("1.100000e+00", printToStringUnbuffered(1.1));

  // void*
  EXPECT_EQ("0x0", printToStringUnbuffered((void*) 0));
  EXPECT_EQ("0xbeef", printToStringUnbuffered((void*) 0xbeef));
  EXPECT_EQ("0xdeadbeef", printToStringUnbuffered((void*) 0xdeadbeef));

  // Min and max.
  EXPECT_EQ("18446744073709551615", printToStringUnbuffered(UINT64_MAX));
  EXPECT_EQ("-9223372036854775808", printToStringUnbuffered(INT64_MIN));
}

TEST(raw_ostreamTest, BufferEdge) {  
  EXPECT_EQ("1.20", printToString(format("%.2f", 1.2), 1));
  EXPECT_EQ("1.20", printToString(format("%.2f", 1.2), 2));
  EXPECT_EQ("1.20", printToString(format("%.2f", 1.2), 3));
  EXPECT_EQ("1.20", printToString(format("%.2f", 1.2), 4));
  EXPECT_EQ("1.20", printToString(format("%.2f", 1.2), 10));
}

TEST(raw_ostreamTest, TinyBuffer) {
  std::string Str;
  raw_string_ostream OS(Str);
  OS.SetBufferSize(1);
  OS << "hello";
  OS << 1;
  OS << 'w' << 'o' << 'r' << 'l' << 'd';
  EXPECT_EQ("hello1world", OS.str());
}

TEST(raw_ostreamTest, WriteEscaped) {
  std::string Str;

  Str = "";
  raw_string_ostream(Str).write_escaped("hi");
  EXPECT_EQ("hi", Str);

  Str = "";
  raw_string_ostream(Str).write_escaped("\\\t\n\"");
  EXPECT_EQ("\\\\\\t\\n\\\"", Str);

  Str = "";
  raw_string_ostream(Str).write_escaped("\1\10\200");
  EXPECT_EQ("\\001\\010\\200", Str);
}

}
