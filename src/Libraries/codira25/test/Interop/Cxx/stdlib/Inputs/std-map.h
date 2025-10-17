/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 25, 2022.
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

#ifndef TEST_INTEROP_CXX_STDLIB_INPUTS_STD_MAP_H
#define TEST_INTEROP_CXX_STDLIB_INPUTS_STD_MAP_H

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

using Map = std::map<int, int>;
using MapStrings = std::map<std::string, std::string>;
using NestedMap = std::map<int, Map>;
using MapGroup = std::map<int, std::vector<int>>;
using UnorderedMap = std::unordered_map<int, int>;
using UnorderedMapGroup = std::unordered_map<int, std::vector<int>>;
inline Map initMap() { return {{1, 3}, {2, 2}, {3, 3}}; }
inline UnorderedMap initUnorderedMap() { return {{1, 3}, {3, 3}, {2, 2}}; }
inline Map initEmptyMap() { return {}; }
inline UnorderedMap initEmptyUnorderedMap() { return {}; }

#endif // TEST_INTEROP_CXX_STDLIB_INPUTS_STD_MAP_H
