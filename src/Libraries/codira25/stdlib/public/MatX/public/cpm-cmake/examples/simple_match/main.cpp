/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 26, 2023.
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

#include <iostream>
#include <simple_match/simple_match.hpp>

int main(int argc, char** argv) {
  using namespace simple_match;
  using namespace simple_match::placeholders;

  std::string input;
  std::cout << "please enter a number or 'quit' to exit" << std::endl;

  while (true) {
    std::cout << "> ";
    std::getline(std::cin, input);
    if (input == "quit") {
      break;
    }
    int x;
    try {
      x = std::stoi(input);
    } catch (std::invalid_argument&) {
      std::cout << "invalid input" << std::endl;
      continue;
    }

    match(
        x, 1, []() { std::cout << "The answer is one\n"; }, 2,
        []() { std::cout << "The answer is two\n"; }, _x < 10,
        [](auto&& a) { std::cout << "The answer " << a << " is less than 10\n"; }, 10 < _x < 20,
        [](auto&& a) { std::cout << "The answer " << a << " is between 10 and 20 exclusive\n"; }, _,
        []() { std::cout << "Did not match\n"; });
  }

  return 0;
}
