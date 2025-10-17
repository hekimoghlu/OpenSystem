/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 27, 2022.
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
#include "test/scenario/column_printer.h"

namespace webrtc {
namespace test {

ColumnPrinter::ColumnPrinter(const ColumnPrinter&) = default;
ColumnPrinter::~ColumnPrinter() = default;

ColumnPrinter::ColumnPrinter(
    const char* headers,
    std::function<void(rtc::SimpleStringBuilder&)> printer,
    size_t max_length)
    : headers_(headers), printer_(printer), max_length_(max_length) {}

ColumnPrinter ColumnPrinter::Fixed(const char* headers, std::string fields) {
  return ColumnPrinter(
      headers, [fields](rtc::SimpleStringBuilder& sb) { sb << fields; },
      fields.size());
}

ColumnPrinter ColumnPrinter::Lambda(
    const char* headers,
    std::function<void(rtc::SimpleStringBuilder&)> printer,
    size_t max_length) {
  return ColumnPrinter(headers, printer, max_length);
}

StatesPrinter::StatesPrinter(std::unique_ptr<RtcEventLogOutput> writer,
                             std::vector<ColumnPrinter> printers)
    : writer_(std::move(writer)), printers_(printers) {
  RTC_CHECK(!printers_.empty());
  for (auto& printer : printers_)
    buffer_size_ += printer.max_length_ + 1;
  buffer_.resize(buffer_size_);
}

StatesPrinter::~StatesPrinter() = default;

void StatesPrinter::PrintHeaders() {
  if (!writer_)
    return;
  writer_->Write(printers_[0].headers_);
  for (size_t i = 1; i < printers_.size(); ++i) {
    writer_->Write(" ");
    writer_->Write(printers_[i].headers_);
  }
  writer_->Write("\n");
}

void StatesPrinter::PrintRow() {
  // Note that this is run for null output to preserve side effects, this allows
  // setting break points etc.
  rtc::SimpleStringBuilder sb(buffer_);
  printers_[0].printer_(sb);
  for (size_t i = 1; i < printers_.size(); ++i) {
    sb << ' ';
    printers_[i].printer_(sb);
  }
  sb << "\n";
  if (writer_)
    writer_->Write(std::string(sb.str(), sb.size()));
}
}  // namespace test
}  // namespace webrtc
