/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 24, 2022.
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

//===----------- VariadicFunctionTest.cpp - VariadicFunction unit tests ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/VariadicFunction.h"

using namespace llvm;
namespace {

// Defines a variadic function StringCat() to join strings.
// StringCat()'s arguments and return value have class types.
std::string StringCatImpl(ArrayRef<const std::string *> Args) {
  std::string S;
  for (unsigned i = 0, e = Args.size(); i < e; ++i)
    S += *Args[i];
  return S;
}
const VariadicFunction<std::string, std::string, StringCatImpl> StringCat = {};

TEST(VariadicFunctionTest, WorksForClassTypes) {
  EXPECT_EQ("", StringCat());
  EXPECT_EQ("a", StringCat("a"));
  EXPECT_EQ("abc", StringCat("a", "bc"));
  EXPECT_EQ("0123456789abcdefghijklmnopqrstuv",
            StringCat("0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                      "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
                      "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
                      "u", "v"));
}

// Defines a variadic function Sum(), whose arguments and return value
// have primitive types.
// The return type of SumImp() is deliberately different from its
// argument type, as we want to test that this works.
long SumImpl(ArrayRef<const int *> Args) {
  long Result = 0;
  for (unsigned i = 0, e = Args.size(); i < e; ++i)
    Result += *Args[i];
  return Result;
}
const VariadicFunction<long, int, SumImpl> Sum = {};

TEST(VariadicFunctionTest, WorksForPrimitiveTypes) {
  EXPECT_EQ(0, Sum());
  EXPECT_EQ(1, Sum(1));
  EXPECT_EQ(12, Sum(10, 2));
  EXPECT_EQ(1234567, Sum(1000000, 200000, 30000, 4000, 500, 60, 7));
}

// Appends an array of strings to dest and returns the number of
// characters appended.
int StringAppendImpl(std::string *Dest, ArrayRef<const std::string *> Args) {
  int Chars = 0;
  for (unsigned i = 0, e = Args.size(); i < e; ++i) {
    Chars += Args[i]->size();
    *Dest += *Args[i];
  }
  return Chars;
}
const VariadicFunction1<int, std::string *, std::string,
                        StringAppendImpl> StringAppend = {};

TEST(VariadicFunction1Test, Works) {
  std::string S0("hi");
  EXPECT_EQ(0, StringAppend(&S0));
  EXPECT_EQ("hi", S0);

  std::string S1("bin");
  EXPECT_EQ(2, StringAppend(&S1, "go"));
  EXPECT_EQ("bingo", S1);

  std::string S4("Fab4");
  EXPECT_EQ(4 + 4 + 6 + 5,
            StringAppend(&S4, "John", "Paul", "George", "Ringo"));
  EXPECT_EQ("Fab4JohnPaulGeorgeRingo", S4);
}

// Counts how many optional arguments fall in the given range.
// Returns the result in *num_in_range.  We make the return type void
// as we want to test that VariadicFunction* can handle it.
void CountInRangeImpl(int *NumInRange, int Low, int High,
                      ArrayRef<const int *> Args) {
  *NumInRange = 0;
  for (unsigned i = 0, e = Args.size(); i < e; ++i)
    if (Low <= *Args[i] && *Args[i] <= High)
      ++(*NumInRange);
}
const VariadicFunction3<void, int *, int, int, int,
                        CountInRangeImpl> CountInRange = {};

TEST(VariadicFunction3Test, Works) {
  int N = -1;
  CountInRange(&N, -100, 100);
  EXPECT_EQ(0, N);

  CountInRange(&N, -100, 100, 42);
  EXPECT_EQ(1, N);

  CountInRange(&N, -100, 100, 1, 999, -200, 42);
  EXPECT_EQ(2, N);
}

}  // namespace
