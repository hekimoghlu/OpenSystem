/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 1, 2022.
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
#ifndef RTC_BASE_EXPERIMENTS_STRUCT_PARAMETERS_PARSER_H_
#define RTC_BASE_EXPERIMENTS_STRUCT_PARAMETERS_PARSER_H_

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/string_view.h"
#include "rtc_base/experiments/field_trial_parser.h"
#include "rtc_base/experiments/field_trial_units.h"
#include "rtc_base/string_encode.h"

namespace webrtc {
namespace struct_parser_impl {
struct TypedMemberParser {
 public:
  bool (*parse)(absl::string_view src, void* target);
  void (*encode)(const void* src, std::string* target);
};

struct MemberParameter {
  const char* key;
  void* member_ptr;
  TypedMemberParser parser;
};

template <typename T>
class TypedParser {
 public:
  static bool Parse(absl::string_view src, void* target);
  static void Encode(const void* src, std::string* target);
};

// Instantiated in cc file to avoid duplication during compile. Add additional
// parsers as needed. Generally, try to use these suggested types even if the
// context where the value is used might require a different type. For instance,
// a size_t representing a packet size should use an int parameter as there's no
// need to support packet sizes larger than INT32_MAX.
extern template class TypedParser<bool>;
extern template class TypedParser<double>;
extern template class TypedParser<int>;
extern template class TypedParser<unsigned>;
extern template class TypedParser<std::optional<double>>;
extern template class TypedParser<std::optional<int>>;
extern template class TypedParser<std::optional<unsigned>>;

extern template class TypedParser<DataRate>;
extern template class TypedParser<DataSize>;
extern template class TypedParser<TimeDelta>;
extern template class TypedParser<std::optional<DataRate>>;
extern template class TypedParser<std::optional<DataSize>>;
extern template class TypedParser<std::optional<TimeDelta>>;

template <typename T>
void AddMembers(MemberParameter* out, const char* key, T* member) {
  *out = MemberParameter{
      key, member,
      TypedMemberParser{&TypedParser<T>::Parse, &TypedParser<T>::Encode}};
}

template <typename T, typename... Args>
void AddMembers(MemberParameter* out,
                const char* key,
                T* member,
                Args... args) {
  AddMembers(out, key, member);
  AddMembers(++out, args...);
}
}  // namespace struct_parser_impl

class StructParametersParser {
 public:
  template <typename T, typename... Args>
  static std::unique_ptr<StructParametersParser> Create(const char* first_key,
                                                        T* first_member,
                                                        Args... args) {
    std::vector<struct_parser_impl::MemberParameter> members(
        sizeof...(args) / 2 + 1);
    struct_parser_impl::AddMembers(&members.front(), std::move(first_key),
                                   first_member, args...);
    return absl::WrapUnique(new StructParametersParser(std::move(members)));
  }

  void Parse(absl::string_view src);
  std::string Encode() const;

 private:
  explicit StructParametersParser(
      std::vector<struct_parser_impl::MemberParameter> members);

  std::vector<struct_parser_impl::MemberParameter> members_;
};

}  // namespace webrtc

#endif  // RTC_BASE_EXPERIMENTS_STRUCT_PARAMETERS_PARSER_H_
