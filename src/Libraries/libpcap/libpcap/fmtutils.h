/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 8, 2023.
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
#ifndef fmtutils_h
#define	fmtutils_h

#include "pcap/funcattrs.h"

#ifdef __cplusplus
extern "C" {
#endif

void	pcap_fmt_set_encoding(unsigned int);

void	pcap_fmt_errmsg_for_errno(char *, size_t, int,
    PCAP_FORMAT_STRING(const char *), ...) PCAP_PRINTFLIKE(4, 5);

#ifdef _WIN32
void	pcap_fmt_errmsg_for_win32_err(char *, size_t, DWORD,
    PCAP_FORMAT_STRING(const char *), ...) PCAP_PRINTFLIKE(4, 5);
#endif

#ifdef __cplusplus
}
#endif

#endif
