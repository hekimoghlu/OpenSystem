/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 26, 2023.
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
#ifndef MODULES_AUDIO_PROCESSING_AEC_DUMP_AEC_DUMP_FACTORY_H_
#define MODULES_AUDIO_PROCESSING_AEC_DUMP_AEC_DUMP_FACTORY_H_

#include <memory>

#include "absl/base/nullability.h"
#include "absl/strings/string_view.h"
#include "api/task_queue/task_queue_base.h"
#include "modules/audio_processing/include/aec_dump.h"
#include "rtc_base/system/file_wrapper.h"
#include "rtc_base/system/rtc_export.h"

namespace webrtc {

class RTC_EXPORT AecDumpFactory {
 public:
  // The `worker_queue` must outlive the created AecDump instance.
  // `max_log_size_bytes == -1` means the log size will be unlimited.
  // The AecDump takes responsibility for `handle` and closes it in the
  // destructor. A non-null return value indicates that the file has been
  // sucessfully opened.
  static absl::Nullable<std::unique_ptr<AecDump>> Create(
      FileWrapper file,
      int64_t max_log_size_bytes,
      absl::Nonnull<TaskQueueBase*> worker_queue);
  static absl::Nullable<std::unique_ptr<AecDump>> Create(
      absl::string_view file_name,
      int64_t max_log_size_bytes,
      absl::Nonnull<TaskQueueBase*> worker_queue);
  static absl::Nullable<std::unique_ptr<AecDump>> Create(
      absl::Nonnull<FILE*> handle,
      int64_t max_log_size_bytes,
      absl::Nonnull<TaskQueueBase*> worker_queue);
};

}  // namespace webrtc

#endif  // MODULES_AUDIO_PROCESSING_AEC_DUMP_AEC_DUMP_FACTORY_H_
