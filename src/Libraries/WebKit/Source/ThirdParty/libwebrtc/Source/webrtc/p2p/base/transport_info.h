/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 2, 2023.
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
#ifndef P2P_BASE_TRANSPORT_INFO_H_
#define P2P_BASE_TRANSPORT_INFO_H_

#include <string>
#include <vector>

#include "api/candidate.h"
#include "p2p/base/p2p_constants.h"
#include "p2p/base/transport_description.h"

namespace cricket {

// A TransportInfo is NOT a transport-info message.  It is comparable
// to a "ContentInfo". A transport-infos message is basically just a
// collection of TransportInfos.
struct TransportInfo {
  TransportInfo() {}

  TransportInfo(const std::string& content_name,
                const TransportDescription& description)
      : content_name(content_name), description(description) {}

  std::string content_name;
  TransportDescription description;
};

typedef std::vector<TransportInfo> TransportInfos;

}  // namespace cricket

#endif  // P2P_BASE_TRANSPORT_INFO_H_
