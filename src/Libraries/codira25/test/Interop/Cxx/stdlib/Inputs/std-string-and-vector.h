/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 5, 2022.
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

#include <string>
#include <vector>

struct Item {
  std::vector<std::string> keys;
  std::vector<std::string> values;
};

inline Item get_item() {
  return {};
}

std::vector<int> makeVecOfInt() { return {1, 2, 3}; }
std::vector<std::string> makeVecOfString() { return {"Hello", "World"}; }

struct S {
private:
  std::string privStr;
  std::vector<std::string> privVec;

public:
  std::string pubStr;
  std::vector<std::string> pubVec;

protected:
  std::string protStr;
  std::vector<std::string> protVec;

public:
  S() : privStr("private"), privVec({"private", "vector"}), 
        pubStr("public"), pubVec({"a", "public", "vector"}), 
        protStr("protected"), protVec({"protected", "vector"}) {}

  std::vector<std::string> getPrivVec() const { return privVec; }
  std::string getProtStr() const { return protStr; }
};
