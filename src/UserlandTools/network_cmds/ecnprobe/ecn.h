/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 13, 2025.
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
void ECNTest (u_int32_t sourceIpAddress, u_int16_t sourcePort, u_int32_t targetIpAddress, u_int16_t targetPort, int mss) ;
void ECNAckData (struct IPPacket *p);
void DataPkt (char *filename, u_int8_t iptos, u_int8_t tcp_flags);
void checkECN ();
void ECNPathCheckTest(u_int32_t sourceIpAddress, u_int16_t surcePort,
  u_int32_t targetIpAddress, u_int16_t targetPort, int mss);
void SynTest(u_int32_t sourceIpAddress, u_int16_t surcePort, u_int32_t targetIpAddress, u_int16_t targetPort, int mss, int syn_reply);
