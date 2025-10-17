/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 24, 2023.
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
#include "api/stats/rtc_stats.h"

#include <cstdio>

#include "rtc_base/strings/string_builder.h"

namespace webrtc {

RTCStats::RTCStats(const RTCStats& other)
    : RTCStats(other.id_, other.timestamp_) {}

RTCStats::~RTCStats() {}

bool RTCStats::operator==(const RTCStats& other) const {
  if (type() != other.type() || id() != other.id())
    return false;
  std::vector<Attribute> attributes = Attributes();
  std::vector<Attribute> other_attributes = other.Attributes();
  RTC_DCHECK_EQ(attributes.size(), other_attributes.size());
  for (size_t i = 0; i < attributes.size(); ++i) {
    if (attributes[i] != other_attributes[i]) {
      return false;
    }
  }
  return true;
}

bool RTCStats::operator!=(const RTCStats& other) const {
  return !(*this == other);
}

std::string RTCStats::ToJson() const {
  rtc::StringBuilder sb;
  sb << "{\"type\":\"" << type()
     << "\","
        "\"id\":\""
     << id_
     << "\","
        "\"timestamp\":"
     << timestamp_.us();
  for (const Attribute& attribute : Attributes()) {
    if (attribute.has_value()) {
      sb << ",\"" << attribute.name() << "\":";
      if (attribute.holds_alternative<std::string>()) {
        sb << "\"";
      }
      sb << attribute.ToString();
      if (attribute.holds_alternative<std::string>()) {
        sb << "\"";
      }
    }
  }
  sb << "}";
  return sb.Release();
}

std::vector<Attribute> RTCStats::Attributes() const {
  return AttributesImpl(0);
}

std::vector<Attribute> RTCStats::AttributesImpl(
    size_t additional_capacity) const {
  std::vector<Attribute> attributes;
  attributes.reserve(additional_capacity);
  return attributes;
}

}  // namespace webrtc
