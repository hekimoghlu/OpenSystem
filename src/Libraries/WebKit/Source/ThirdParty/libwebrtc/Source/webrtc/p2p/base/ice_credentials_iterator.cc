/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 5, 2025.
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
#include "p2p/base/ice_credentials_iterator.h"

#include "p2p/base/p2p_constants.h"
#include "rtc_base/crypto_random.h"

namespace cricket {

IceCredentialsIterator::IceCredentialsIterator(
    const std::vector<IceParameters>& pooled_credentials)
    : pooled_ice_credentials_(pooled_credentials) {}

IceCredentialsIterator::~IceCredentialsIterator() = default;

IceParameters IceCredentialsIterator::CreateRandomIceCredentials() {
  return IceParameters(rtc::CreateRandomString(ICE_UFRAG_LENGTH),
                       rtc::CreateRandomString(ICE_PWD_LENGTH), false);
}

IceParameters IceCredentialsIterator::GetIceCredentials() {
  if (pooled_ice_credentials_.empty()) {
    return CreateRandomIceCredentials();
  }
  IceParameters credentials = pooled_ice_credentials_.back();
  pooled_ice_credentials_.pop_back();
  return credentials;
}

}  // namespace cricket
