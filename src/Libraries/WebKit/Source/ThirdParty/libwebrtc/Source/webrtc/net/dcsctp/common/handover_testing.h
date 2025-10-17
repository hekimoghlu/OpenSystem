/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 3, 2023.
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
#ifndef NET_DCSCTP_COMMON_HANDOVER_TESTING_H_
#define NET_DCSCTP_COMMON_HANDOVER_TESTING_H_

#include "net/dcsctp/public/dcsctp_handover_state.h"

namespace dcsctp {
// This global function is to facilitate testing of the socket handover state
// (`DcSctpSocketHandoverState`) serialization. dcSCTP library users have to
// implement state serialization if it's needed. To test the serialization one
// can set a custom `g_handover_state_transformer_for_test` at startup, link to
// the dcSCTP tests and run the resulting binary. Custom function can serialize
// and deserialize the passed state. All dcSCTP handover tests call
// `g_handover_state_transformer_for_test`. If some part of the state is
// serialized incorrectly or is forgotten, high chance that it will fail the
// tests.
extern void (*g_handover_state_transformer_for_test)(
    DcSctpSocketHandoverState*);
}  // namespace dcsctp

#endif  // NET_DCSCTP_COMMON_HANDOVER_TESTING_H_
