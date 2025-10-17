/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 8, 2023.
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
#ifndef TEST_SCENARIO_COLUMN_PRINTER_H_
#define TEST_SCENARIO_COLUMN_PRINTER_H_
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "rtc_base/strings/string_builder.h"
#include "test/logging/log_writer.h"

namespace webrtc {
namespace test {
class ColumnPrinter {
 public:
  ColumnPrinter(const ColumnPrinter&);
  ~ColumnPrinter();
  static ColumnPrinter Fixed(const char* headers, std::string fields);
  static ColumnPrinter Lambda(
      const char* headers,
      std::function<void(rtc::SimpleStringBuilder&)> printer,
      size_t max_length = 256);

 protected:
  friend class StatesPrinter;
  const char* headers_;
  std::function<void(rtc::SimpleStringBuilder&)> printer_;
  size_t max_length_;

 private:
  ColumnPrinter(const char* headers,
                std::function<void(rtc::SimpleStringBuilder&)> printer,
                size_t max_length);
};

class StatesPrinter {
 public:
  StatesPrinter(std::unique_ptr<RtcEventLogOutput> writer,
                std::vector<ColumnPrinter> printers);

  ~StatesPrinter();

  StatesPrinter(const StatesPrinter&) = delete;
  StatesPrinter& operator=(const StatesPrinter&) = delete;

  void PrintHeaders();
  void PrintRow();

 private:
  const std::unique_ptr<RtcEventLogOutput> writer_;
  const std::vector<ColumnPrinter> printers_;
  size_t buffer_size_ = 0;
  std::vector<char> buffer_;
};
}  // namespace test
}  // namespace webrtc

#endif  // TEST_SCENARIO_COLUMN_PRINTER_H_
