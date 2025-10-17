/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 13, 2024.
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

//
//  SOSPeerOTRTimer.h
//

#ifndef SOSPeerOTRTimer_h
#define SOSPeerOTRTimer_h

void SOSPeerOTRTimerFired(SOSAccount* account, SOSPeerRef peer, SOSEngineRef engine, SOSCoderRef coder);
int SOSPeerOTRTimerTimeoutValue(SOSAccount* account, SOSPeerRef peer);
void SOSPeerOTRTimerSetupAwaitingTimer(SOSAccount* account, SOSPeerRef peer, SOSEngineRef engine, SOSCoderRef coder);


//functions to handle max retry counter
void SOSPeerOTRTimerIncreaseOTRNegotiationRetryCount(SOSAccount* account, NSString* peerid);
bool SOSPeerOTRTimerHaveReachedMaxRetryAllowance(SOSAccount* account, NSString* peerid);
void SOSPeerOTRTimerClearMaxRetryCount(SOSAccount* account, NSString* peerid);
bool SOSPeerOTRTimerHaveAnRTTAvailable(SOSAccount* account, NSString* peerid);
void SOSPeerOTRTimerRemoveRTTTimeoutForPeer(SOSAccount* account, NSString* peerid);
void SOSPeerOTRTimerRemoveLastSentMessageTimestamp(SOSAccount* account, NSString* peerid);

#endif /* SOSPeerOTRTimer_h */
