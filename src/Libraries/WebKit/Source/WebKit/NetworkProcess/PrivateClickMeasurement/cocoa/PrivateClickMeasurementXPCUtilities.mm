/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 21, 2023.
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
#import "config.h"
#import "PrivateClickMeasurementXPCUtilities.h"

#import "DaemonUtilities.h"
#import "PrivateClickMeasurementManagerInterface.h"
#import <wtf/OSObjectPtr.h>

namespace WebKit::PCM {

void addVersionAndEncodedMessageToDictionary(Vector<uint8_t>&& message, xpc_object_t dictionary)
{
    ASSERT(xpc_get_type(dictionary) == XPC_TYPE_DICTIONARY);
    xpc_dictionary_set_uint64(dictionary, PCM::protocolVersionKey, PCM::protocolVersionValue);
    xpc_dictionary_set_value(dictionary, PCM::protocolEncodedMessageKey, vectorToXPCData(WTFMove(message)).get());
}

} // namespace WebKit
