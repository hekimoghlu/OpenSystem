/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 1, 2024.
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
#include "net/dcsctp/packet/error_cause/error_cause.h"

#include <stddef.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "api/array_view.h"
#include "net/dcsctp/common/math.h"
#include "net/dcsctp/packet/error_cause/cookie_received_while_shutting_down_cause.h"
#include "net/dcsctp/packet/error_cause/invalid_mandatory_parameter_cause.h"
#include "net/dcsctp/packet/error_cause/invalid_stream_identifier_cause.h"
#include "net/dcsctp/packet/error_cause/missing_mandatory_parameter_cause.h"
#include "net/dcsctp/packet/error_cause/no_user_data_cause.h"
#include "net/dcsctp/packet/error_cause/out_of_resource_error_cause.h"
#include "net/dcsctp/packet/error_cause/protocol_violation_cause.h"
#include "net/dcsctp/packet/error_cause/restart_of_an_association_with_new_address_cause.h"
#include "net/dcsctp/packet/error_cause/stale_cookie_error_cause.h"
#include "net/dcsctp/packet/error_cause/unrecognized_chunk_type_cause.h"
#include "net/dcsctp/packet/error_cause/unrecognized_parameter_cause.h"
#include "net/dcsctp/packet/error_cause/unresolvable_address_cause.h"
#include "net/dcsctp/packet/error_cause/user_initiated_abort_cause.h"
#include "rtc_base/strings/string_builder.h"

namespace dcsctp {

template <class ErrorCause>
bool ParseAndPrint(ParameterDescriptor descriptor, rtc::StringBuilder& sb) {
  if (descriptor.type == ErrorCause::kType) {
    std::optional<ErrorCause> p = ErrorCause::Parse(descriptor.data);
    if (p.has_value()) {
      sb << p->ToString();
    } else {
      sb << "Failed to parse error cause of type " << ErrorCause::kType;
    }
    return true;
  }
  return false;
}

std::string ErrorCausesToString(const Parameters& parameters) {
  rtc::StringBuilder sb;

  std::vector<ParameterDescriptor> descriptors = parameters.descriptors();
  for (size_t i = 0; i < descriptors.size(); ++i) {
    if (i > 0) {
      sb << "\n";
    }

    const ParameterDescriptor& d = descriptors[i];
    if (!ParseAndPrint<InvalidStreamIdentifierCause>(d, sb) &&
        !ParseAndPrint<MissingMandatoryParameterCause>(d, sb) &&
        !ParseAndPrint<StaleCookieErrorCause>(d, sb) &&
        !ParseAndPrint<OutOfResourceErrorCause>(d, sb) &&
        !ParseAndPrint<UnresolvableAddressCause>(d, sb) &&
        !ParseAndPrint<UnrecognizedChunkTypeCause>(d, sb) &&
        !ParseAndPrint<InvalidMandatoryParameterCause>(d, sb) &&
        !ParseAndPrint<UnrecognizedParametersCause>(d, sb) &&
        !ParseAndPrint<NoUserDataCause>(d, sb) &&
        !ParseAndPrint<CookieReceivedWhileShuttingDownCause>(d, sb) &&
        !ParseAndPrint<RestartOfAnAssociationWithNewAddressesCause>(d, sb) &&
        !ParseAndPrint<UserInitiatedAbortCause>(d, sb) &&
        !ParseAndPrint<ProtocolViolationCause>(d, sb)) {
      sb << "Unhandled parameter of type: " << d.type;
    }
  }

  return sb.Release();
}
}  // namespace dcsctp
