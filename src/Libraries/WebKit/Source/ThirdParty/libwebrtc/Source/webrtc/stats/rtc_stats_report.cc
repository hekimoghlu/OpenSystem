/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 12, 2021.
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
#include "api/stats/rtc_stats_report.h"

#include <type_traits>
#include <utility>

#include "rtc_base/checks.h"
#include "rtc_base/strings/string_builder.h"

namespace webrtc {

RTCStatsReport::ConstIterator::ConstIterator(
    const rtc::scoped_refptr<const RTCStatsReport>& report,
    StatsMap::const_iterator it)
    : report_(report), it_(it) {}

RTCStatsReport::ConstIterator::ConstIterator(ConstIterator&& other) = default;

RTCStatsReport::ConstIterator::~ConstIterator() {}

RTCStatsReport::ConstIterator& RTCStatsReport::ConstIterator::operator++() {
  ++it_;
  return *this;
}

RTCStatsReport::ConstIterator& RTCStatsReport::ConstIterator::operator++(int) {
  return ++(*this);
}

const RTCStats& RTCStatsReport::ConstIterator::operator*() const {
  return *it_->second.get();
}

const RTCStats* RTCStatsReport::ConstIterator::operator->() const {
  return it_->second.get();
}

bool RTCStatsReport::ConstIterator::operator==(
    const RTCStatsReport::ConstIterator& other) const {
  return it_ == other.it_;
}

bool RTCStatsReport::ConstIterator::operator!=(
    const RTCStatsReport::ConstIterator& other) const {
  return !(*this == other);
}

rtc::scoped_refptr<RTCStatsReport> RTCStatsReport::Create(Timestamp timestamp) {
  return rtc::scoped_refptr<RTCStatsReport>(new RTCStatsReport(timestamp));
}

RTCStatsReport::RTCStatsReport(Timestamp timestamp) : timestamp_(timestamp) {}

rtc::scoped_refptr<RTCStatsReport> RTCStatsReport::Copy() const {
  rtc::scoped_refptr<RTCStatsReport> copy = Create(timestamp_);
  for (auto it = stats_.begin(); it != stats_.end(); ++it) {
    copy->AddStats(it->second->copy());
  }
  return copy;
}

void RTCStatsReport::AddStats(std::unique_ptr<const RTCStats> stats) {
#if RTC_DCHECK_IS_ON
  auto result =
#endif
      stats_.insert(std::make_pair(std::string(stats->id()), std::move(stats)));
#if RTC_DCHECK_IS_ON
  RTC_DCHECK(result.second)
      << "A stats object with ID \"" << result.first->second->id() << "\" is "
      << "already present in this stats report.";
#endif
}

const RTCStats* RTCStatsReport::Get(const std::string& id) const {
  StatsMap::const_iterator it = stats_.find(id);
  if (it != stats_.cend())
    return it->second.get();
  return nullptr;
}

std::unique_ptr<const RTCStats> RTCStatsReport::Take(const std::string& id) {
  StatsMap::iterator it = stats_.find(id);
  if (it == stats_.end())
    return nullptr;
  std::unique_ptr<const RTCStats> stats = std::move(it->second);
  stats_.erase(it);
  return stats;
}

void RTCStatsReport::TakeMembersFrom(rtc::scoped_refptr<RTCStatsReport> other) {
  for (StatsMap::iterator it = other->stats_.begin(); it != other->stats_.end();
       ++it) {
    AddStats(std::unique_ptr<const RTCStats>(it->second.release()));
  }
  other->stats_.clear();
}

RTCStatsReport::ConstIterator RTCStatsReport::begin() const {
  return ConstIterator(rtc::scoped_refptr<const RTCStatsReport>(this),
                       stats_.cbegin());
}

RTCStatsReport::ConstIterator RTCStatsReport::end() const {
  return ConstIterator(rtc::scoped_refptr<const RTCStatsReport>(this),
                       stats_.cend());
}

std::string RTCStatsReport::ToJson() const {
  if (begin() == end()) {
    return "";
  }
  rtc::StringBuilder sb;
  sb << "[";
  const char* separator = "";
  for (ConstIterator it = begin(); it != end(); ++it) {
    sb << separator << it->ToJson();
    separator = ",";
  }
  sb << "]";
  return sb.Release();
}

}  // namespace webrtc
