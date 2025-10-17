/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 13, 2025.
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
#include <ctype.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <string>
#include <vector>

#include <android-base/file.h>
#include <android-base/logging.h>
#include <android-base/strings.h>
#include <benchmark/benchmark.h>
#include <property_info_parser/property_info_parser.h>
#include <property_info_serializer/property_info_serializer.h>
#include <system_properties/contexts_split.h>

#include "context_lookup_benchmark_data.h"

using android::base::Split;
using android::base::WriteStringToFd;
using android::properties::BuildTrie;
using android::properties::ParsePropertyInfoFile;
using android::properties::PropertyInfoArea;
using android::properties::PropertyInfoEntry;

BENCHMARK_MAIN();

class LegacyPropertyMapping : public ContextsSplit {
 public:
  LegacyPropertyMapping(const char* property_contexts) {
    TemporaryFile file;
    if (!WriteStringToFd(property_contexts, file.fd)) {
      PLOG(FATAL) << "Could not write to temporary file";
    }

    if (!InitializePropertiesFromFile(file.path)) {
      LOG(FATAL) << "Could not initialize properties";
    }
  }
};

static std::vector<std::string> PropertiesToLookup() {
  std::vector<std::string> properties;
  auto property_lines = Split(aosp_s_property_contexts, "\n");
  for (const auto& line : property_lines) {
    if (line.empty() || line[0] == '#') {
      continue;
    }

    auto property = Split(line, " ")[0];
    properties.push_back(property);
    properties.push_back(property + "0");
    properties.push_back(property + "A");
  }
  return properties;
}

static void LegacyLookupOreo(benchmark::State& state) {
  LegacyPropertyMapping mapping(oreo_property_contexts);
  auto properties = PropertiesToLookup();
  for (auto _ : state) {
    for (const auto& property : properties) {
      benchmark::DoNotOptimize(mapping.GetPrefixNodeForName(property.c_str()));
    }
  }
}
BENCHMARK(LegacyLookupOreo);

static void LegacyLookupS(benchmark::State& state) {
  LegacyPropertyMapping mapping(aosp_s_property_contexts);
  auto properties = PropertiesToLookup();
  for (auto _ : state) {
    for (const auto& property : properties) {
      benchmark::DoNotOptimize(mapping.GetPrefixNodeForName(property.c_str()));
    }
  }
}
BENCHMARK(LegacyLookupS);

static std::string CreateSerializedTrie(const char* input_file) {
  std::vector<std::string> errors;
  std::vector<PropertyInfoEntry> property_infos;
  ParsePropertyInfoFile(input_file, false, &property_infos, &errors);

  std::string serialized_trie;
  std::string error;
  if (!BuildTrie(property_infos, "u:object_r:default_prop:s0", "string", &serialized_trie,
                 &error)) {
    LOG(FATAL) << "Could not build trie: " << error;
  }
  return serialized_trie;
}

static void TrieLookupOreo(benchmark::State& state) {
  std::string serialized_trie = CreateSerializedTrie(oreo_property_contexts);
  PropertyInfoArea* trie = reinterpret_cast<PropertyInfoArea*>(serialized_trie.data());
  auto properties = PropertiesToLookup();
  for (auto _ : state) {
    for (const auto& property : properties) {
      trie->GetPropertyInfo(property.c_str(), nullptr, nullptr);
    }
  }
}
BENCHMARK(TrieLookupOreo);

static void TrieLookupS(benchmark::State& state) {
  std::string serialized_trie = CreateSerializedTrie(aosp_s_property_contexts);
  PropertyInfoArea* trie = reinterpret_cast<PropertyInfoArea*>(serialized_trie.data());
  auto properties = PropertiesToLookup();
  for (auto _ : state) {
    for (const auto& property : properties) {
      trie->GetPropertyInfo(property.c_str(), nullptr, nullptr);
    }
  }
}
BENCHMARK(TrieLookupS);
