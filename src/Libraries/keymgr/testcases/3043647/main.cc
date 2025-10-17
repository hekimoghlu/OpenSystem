/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 9, 2024.
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
#include <stdexcept>
#include <string>

#include <iostream>

void g(int n)
{
  if (n == 17) {
    std::cerr << "Throwing an exception" << std::endl;
    throw std::invalid_argument("17 is a bad number");
  }
  std::cerr << "Not throwing an exception" << std::endl;
}

int do_test_2(int n)
{
  std::cerr << "Entering do_test_2\n";
  try {
    g(n);
  }
  catch(const std::invalid_argument& e) {
    std::cerr << "Caught an exception: " << e.what() << std::endl;
    return 1;
  }
  catch(...) {
    std::cerr << "Caught an unexpected exception" << std::endl;
    return 2;
  }

  std::cerr << "Normal exit from do_test_2\n";
  return 0;
}

extern "C" int do_test(int n);

int main()
{
  std::cerr << "About to call do_test" << std::endl;
  int result = do_test(16);
  std::cerr << "Return value is " << result << std::endl;

  std::cerr << "About to call do_test_2" << std::endl;
  int result2 = do_test_2(17);
  std::cerr << "Return value is " << result2 << std::endl;
}
