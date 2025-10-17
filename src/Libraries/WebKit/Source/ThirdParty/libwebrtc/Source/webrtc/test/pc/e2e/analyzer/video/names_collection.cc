/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 29, 2023.
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
#include "test/pc/e2e/analyzer/video/names_collection.h"

#include <optional>
#include <set>

#include "absl/strings/string_view.h"

namespace webrtc {

NamesCollection::NamesCollection(rtc::ArrayView<const std::string> names) {
  names_ = std::vector<std::string>(names.begin(), names.end());
  for (size_t i = 0; i < names_.size(); ++i) {
    index_.emplace(names_[i], i);
    removed_.emplace_back(false);
  }
  size_ = names_.size();
}

bool NamesCollection::HasName(absl::string_view name) const {
  auto it = index_.find(name);
  if (it == index_.end()) {
    return false;
  }
  return !removed_[it->second];
}

size_t NamesCollection::AddIfAbsent(absl::string_view name) {
  auto it = index_.find(name);
  if (it != index_.end()) {
    // Name was registered in the collection before: we need to restore it.
    size_t index = it->second;
    if (removed_[index]) {
      removed_[index] = false;
      size_++;
    }
    return index;
  }
  size_t out = names_.size();
  size_t old_capacity = names_.capacity();
  names_.emplace_back(name);
  removed_.emplace_back(false);
  size_++;
  size_t new_capacity = names_.capacity();

  if (old_capacity == new_capacity) {
    index_.emplace(names_[out], out);
  } else {
    // Reallocation happened in the vector, so we need to rebuild `index_` to
    // fix absl::string_view internal references.
    index_.clear();
    for (size_t i = 0; i < names_.size(); ++i) {
      index_.emplace(names_[i], i);
    }
  }
  return out;
}

std::optional<size_t> NamesCollection::RemoveIfPresent(absl::string_view name) {
  auto it = index_.find(name);
  if (it == index_.end()) {
    return std::nullopt;
  }
  size_t index = it->second;
  if (removed_[index]) {
    return std::nullopt;
  }
  removed_[index] = true;
  size_--;
  return index;
}

std::set<size_t> NamesCollection::GetPresentIndexes() const {
  std::set<size_t> out;
  for (size_t i = 0; i < removed_.size(); ++i) {
    if (!removed_[i]) {
      out.insert(i);
    }
  }
  return out;
}

std::set<size_t> NamesCollection::GetAllIndexes() const {
  std::set<size_t> out;
  for (size_t i = 0; i < names_.size(); ++i) {
    out.insert(i);
  }
  return out;
}

}  // namespace webrtc
