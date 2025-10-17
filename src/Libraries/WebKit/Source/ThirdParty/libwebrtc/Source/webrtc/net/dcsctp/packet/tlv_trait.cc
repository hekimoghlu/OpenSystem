/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 18, 2022.
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
#include "net/dcsctp/packet/tlv_trait.h"

#include "rtc_base/logging.h"

namespace dcsctp {
namespace tlv_trait_impl {
void ReportInvalidSize(size_t actual_size, size_t expected_size) {
  RTC_DLOG(LS_WARNING) << "Invalid size (" << actual_size
                       << ", expected minimum " << expected_size << " bytes)";
}

void ReportInvalidType(int actual_type, int expected_type) {
  RTC_DLOG(LS_WARNING) << "Invalid type (" << actual_type << ", expected "
                       << expected_type << ")";
}

void ReportInvalidFixedLengthField(size_t value, size_t expected) {
  RTC_DLOG(LS_WARNING) << "Invalid length field (" << value << ", expected "
                       << expected << " bytes)";
}

void ReportInvalidVariableLengthField(size_t value, size_t available) {
  RTC_DLOG(LS_WARNING) << "Invalid length field (" << value << ", available "
                       << available << " bytes)";
}

void ReportInvalidPadding(size_t padding_bytes) {
  RTC_DLOG(LS_WARNING) << "Invalid padding (" << padding_bytes << " bytes)";
}

void ReportInvalidLengthMultiple(size_t length, size_t alignment) {
  RTC_DLOG(LS_WARNING) << "Invalid length field (" << length
                       << ", expected an even multiple of " << alignment
                       << " bytes)";
}
}  // namespace tlv_trait_impl
}  // namespace dcsctp
