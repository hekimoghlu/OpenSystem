/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 12, 2024.
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
#ifndef _PARSE_URL_H_
#define _PARSE_URL_H_

#define SMB_SCHEME_STRING  "smb"

int isUrlStringEqual(CFURLRef url1, CFURLRef url2);
char *CStringCreateWithCFString(CFStringRef inStr);
void CreateSMBFromName(struct smb_ctx *ctx, char *fromname, int maxlen);
int isBTMMAddress(CFStringRef serverNameRef);
int ParseSMBURL(struct smb_ctx *ctx);
CFURLRef CreateURLFromReferral(CFStringRef inStr);
CFURLRef CreateSMBURL(const char *url);
int smb_url_to_dictionary(CFURLRef url, CFDictionaryRef *dict);
int smb_dictionary_to_url(CFDictionaryRef dict, CFURLRef *url);
CFStringRef CreateURLCFString(CFStringRef Domain, CFStringRef Username, 
							  CFStringRef Password, CFStringRef ServerName, 
							  CFStringRef Path, CFStringRef PortNumber);

#endif /* _PARSE_URL_H_ */
