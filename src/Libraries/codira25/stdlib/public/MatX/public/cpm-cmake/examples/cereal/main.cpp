/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 4, 2023.
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

#define CATCH_CONFIG_MAIN

#include <cereal/archives/json.hpp>
#include <cereal/cereal.hpp>
#include <sstream>
#include <string>

struct player_data {
  int id{-1};
  std::string name{};
};

template <typename Archive> void serialize(Archive &archive, player_data const &data) {
  archive(cereal::make_nvp("id", data.id), cereal::make_nvp("name", data.name));
}

int main(int argc, char const *argv[]) {
  player_data player{3, "Gamer One"};
  std::ostringstream oss;
  cereal::JSONOutputArchive output(oss);
  output(cereal::make_nvp("player_data", player));

  std::cout << oss.str() << std::endl;

  return 0;
}
