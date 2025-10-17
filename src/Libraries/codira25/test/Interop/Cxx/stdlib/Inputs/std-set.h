/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 22, 2024.
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

#ifndef TEST_INTEROP_CXX_STDLIB_INPUTS_STD_SET_H
#define TEST_INTEROP_CXX_STDLIB_INPUTS_STD_SET_H

#include <set>
#include <unordered_set>

using SetOfCInt = std::set<int>;
using UnorderedSetOfCInt = std::unordered_set<int>;
using MultisetOfCInt = std::multiset<int>;

inline SetOfCInt initSetOfCInt() { return {1, 5, 3}; }
inline UnorderedSetOfCInt initUnorderedSetOfCInt() { return {2, 4, 6}; }
inline MultisetOfCInt initMultisetOfCInt() { return {2, 2, 4, 6}; }

#endif // TEST_INTEROP_CXX_STDLIB_INPUTS_STD_SET_H
