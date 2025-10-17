/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 3, 2021.
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
#ifdef SMB_DEBUG
int testGettingDfsReferralDict(struct smb_ctx * ctx, const char *referral);
#endif // SMB_DEBUG

int checkForDfsReferral(struct smb_ctx * in_ctx, struct smb_ctx ** out_ctx,
                        CFMutableArrayRef dfsReferralDictArray);
int decodeDfsReferral(struct smb_ctx *inConn, mdchain_t mdp,
                      char *rcv_buffer, uint32_t rcv_buffer_len,
                      const char *dfs_referral_str,
                      CFMutableDictionaryRef *outReferralDict);
int getDfsReferralDict(struct smb_ctx * inConn, CFStringRef referralStr,
                       uint16_t maxReferralVersion, CFMutableDictionaryRef *outReferralDict);
int getDfsReferralList(struct smb_ctx * inConn, CFMutableDictionaryRef dfsReferralDict);
