/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 29, 2024.
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
#ifndef TEST_PC_E2E_ANALYZER_VIDEO_NAMES_COLLECTION_H_
#define TEST_PC_E2E_ANALYZER_VIDEO_NAMES_COLLECTION_H_

#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "api/array_view.h"

namespace webrtc {

// Contains mapping between string names and unique size_t values (indexes).
// Once the name is added to the collection it is guaranteed:
//   1. Name will have the same index until collection will be destructed
//   2. Adding, removing and re-adding name won't change its index
//
// The name is considered in the collection if it was added and wasn't removed.
// Adding the name when it is in the collection won't change the collection, the
// same as removing the name when it is removed.
//
// Collection will return name's index and name for the index independently from
// was name removed or not. Once the name was added to the collection the index
// will be allocated for it. To check if name is in collection right now user
// has to explicitly call to `HasName` function.
class NamesCollection {
 public:
  NamesCollection() = default;

  explicit NamesCollection(rtc::ArrayView<const std::string> names);

  // Returns amount of currently presented names in the collection.
  size_t size() const { return size_; }

  // Returns amount of all names known to this collection.
  size_t GetKnownSize() const { return names_.size(); }

  // Returns index of the `name` which was known to the collection. Crashes
  // if `name` was never registered in the collection.
  size_t index(absl::string_view name) const { return index_.at(name); }

  // Returns name which was known to the collection for the specified `index`.
  // Crashes if there was no any name registered in the collection for such
  // `index`.
  const std::string& name(size_t index) const { return names_.at(index); }

  // Returns if `name` is currently presented in this collection.
  bool HasName(absl::string_view name) const;

  // Adds specified `name` to the collection if it isn't presented.
  // Returns index which corresponds to specified `name`.
  size_t AddIfAbsent(absl::string_view name);

  // Removes specified `name` from the collection if it is presented.
  //
  // After name was removed, collection size will be decreased, but `name` index
  // will be preserved. Collection will return false for `HasName(name)`, but
  // will continue to return previously known index for `index(name)` and return
  // `name` for `name(index(name))`.
  //
  // Returns the index of the removed value or std::nullopt if no such `name`
  // registered in the collection.
  std::optional<size_t> RemoveIfPresent(absl::string_view name);

  // Returns a set of indexes for all currently present names in the
  // collection.
  std::set<size_t> GetPresentIndexes() const;

  // Returns a set of all indexes known to the collection including indexes for
  // names that were removed.
  std::set<size_t> GetAllIndexes() const;

 private:
  std::vector<std::string> names_;
  std::vector<bool> removed_;
  std::map<absl::string_view, size_t> index_;
  size_t size_ = 0;
};

}  // namespace webrtc

#endif  // TEST_PC_E2E_ANALYZER_VIDEO_NAMES_COLLECTION_H_
