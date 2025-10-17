/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 19, 2022.
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
#ifndef sec_SecOTRRemote_h
#define sec_SecOTRRemote_h

#include <CoreFoundation/CoreFoundation.h>

CFDataRef SecOTRSessionCreateRemote_internal(CFDataRef publicAccountData, CFDataRef publicPeerId, CFDataRef privateAccountData, CFErrorRef *error);
CFDataRef _SecOTRSessionCreateRemote(CFDataRef publicPeerId, CFErrorRef *error);
bool _SecOTRSessionProcessPacketRemote(CFDataRef sessionData, CFDataRef inputPacket, CFDataRef* outputSessionData, CFDataRef* outputPacket, bool *readyForMessages, CFErrorRef *error);

#endif
