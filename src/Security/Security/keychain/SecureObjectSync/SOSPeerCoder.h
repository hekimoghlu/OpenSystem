/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 24, 2023.
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
#ifndef SOSPeerCoder_h
#define SOSPeerCoder_h
#include "keychain/SecureObjectSync/SOSTransportMessage.h"
#include "keychain/SecureObjectSync/SOSCoder.h"
#include "keychain/SecureObjectSync/SOSEngine.h"
enum SOSCoderUnwrapStatus{
    SOSCoderUnwrapError = 0,
    SOSCoderUnwrapDecoded = 1,
    SOSCoderUnwrapHandled = 2
};

bool SOSPeerCoderSendMessageIfNeeded(SOSAccount* account, SOSEngineRef engine, SOSTransactionRef txn, SOSPeerRef peer, SOSCoderRef coder, CFDataRef *message_to_send, CFStringRef peer_id, CFMutableArrayRef *attributeList, SOSEnginePeerMessageCallBackInfo **sentCallback, CFErrorRef *error);

enum SOSCoderUnwrapStatus SOSPeerHandleCoderMessage(SOSPeerRef peer, SOSCoderRef coder, CFStringRef peer_id, CFDataRef codedMessage, CFDataRef *decodedMessage, bool *forceSave, CFErrorRef *error);

bool SOSPeerSendMessageIfNeeded(SOSPeerRef peer, CFDataRef *message, CFDataRef *message_to_send, SOSCoderRef *coder, CFStringRef circle_id, CFStringRef peer_id, SOSEnginePeerMessageCallBackInfo **sentCallback, CFErrorRef *error);

#endif
