/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 23, 2022.
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

#include "swift/Basic/SuccessorMap.h"
#include "toolchain/Support/raw_ostream.h"
#include <map>
#include <random>

const unsigned RandomSpread = 10;

int main(int argc, char **argv) {
  std::random_device randomDevice; // used for seeding
  std::default_random_engine generator(randomDevice());
  std::uniform_int_distribution<unsigned> distribution(0,RandomSpread);

  swift::SuccessorMap<unsigned, unsigned> map;
  std::map<unsigned, unsigned> stdMap;

  if (argc < 0) map.dump(); // force this to be used

  auto next = [&] { return distribution(generator); };
  auto nextUnmappedKey = [&] {
    unsigned key;
    do {
      key = next();
    } while (stdMap.find(key) != stdMap.end());
    return key;
  };

  while (true) {
    auto operation = next();

    // Find.
    if (operation >= .7 * RandomSpread) {
      unsigned key = nextUnmappedKey();
      auto iter = stdMap.upper_bound(key);
      auto stdResult = (iter == stdMap.end() ? nullptr : &iter->second);

      toolchain::outs() << "  EXPECT_EQ(";
      if (stdResult) {
        toolchain::outs() << *stdResult << ", *";
      } else {
        toolchain::outs() << "InvalidValue, ";
      }
      toolchain::outs() << "map.findLeastUpperBound(" << key << "));\n";

      auto result = map.findLeastUpperBound(key);
      if (result && stdResult && *result != *stdResult) {
        toolchain::outs() << "FAILURE: found " << *result
                     << ", but should have found " << *stdResult << "\n";
        abort();
      } else if (!result && stdResult) {
        toolchain::outs() << "FAILURE: found nothing, but should have found "
                     << *stdResult << "\n";
        abort();
      } else if (result && !stdResult) {
        toolchain::outs() << "FAILURE: found " << *result
                     << ", but should have found nothing\n";
        abort();
      }

    } else if (operation >= .05 * RandomSpread) {
      unsigned key = nextUnmappedKey();
      unsigned value = next();

      toolchain::outs() << "  map.insert(" << key << ", " << value << ");\n";

      map.insert((unsigned) key, (unsigned) value);
      stdMap.insert(std::make_pair(key, value));
    } else {
      toolchain::outs() << "  map.clear();\n";
      map.clear();
      stdMap.clear();
    }

    toolchain::outs() << "  map.validate();\n";
    map.validate();
  }
}
