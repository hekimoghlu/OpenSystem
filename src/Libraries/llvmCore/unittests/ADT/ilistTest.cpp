/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 13, 2024.
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

//===- llvm/unittest/ADT/APInt.cpp - APInt unit tests ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <ostream>
#include "gtest/gtest.h"
#include "llvm/ADT/ilist.h"
#include "llvm/ADT/ilist_node.h"

using namespace llvm;

namespace {

struct Node : ilist_node<Node> {
  int Value;

  Node() {}
  Node(int _Value) : Value(_Value) {}
};

TEST(ilistTest, Basic) {
  ilist<Node> List;
  List.push_back(Node(1));
  EXPECT_EQ(1, List.back().Value);
  EXPECT_EQ(0, List.back().getPrevNode());
  EXPECT_EQ(0, List.back().getNextNode());

  List.push_back(Node(2));
  EXPECT_EQ(2, List.back().Value);
  EXPECT_EQ(2, List.front().getNextNode()->Value);
  EXPECT_EQ(1, List.back().getPrevNode()->Value);

  const ilist<Node> &ConstList = List;
  EXPECT_EQ(2, ConstList.back().Value);
  EXPECT_EQ(2, ConstList.front().getNextNode()->Value);
  EXPECT_EQ(1, ConstList.back().getPrevNode()->Value);
}

}
