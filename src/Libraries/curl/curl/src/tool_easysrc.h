/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 2, 2022.
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

#ifndef HEADER_CURL_TOOL_EASYSRC_H
#define HEADER_CURL_TOOL_EASYSRC_H
/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) Daniel Stenberg, <daniel@haxx.se>, et al.
 *
 * This software is licensed as described in the file COPYING, which
 * you should have received as part of this distribution. The terms
 * are also available at https://curl.se/docs/copyright.html.
 *
 * You may opt to use, copy, modify, merge, publish, distribute and/or sell
 * copies of the Software, and permit persons to whom the Software is
 * furnished to do so, under the terms of the COPYING file.
 *
 * This software is distributed on an "AS IS" basis, WITHOUT WARRANTY OF ANY
 * KIND, either express or implied.
 *
 * SPDX-License-Identifier: curl
 *
 ***************************************************************************/
#include "tool_setup.h"
#ifndef CURL_DISABLE_LIBCURL_OPTION

/* global variable declarations, for easy-interface source code generation */

extern struct slist_wc *easysrc_decl; /* Variable declarations */
extern struct slist_wc *easysrc_data; /* Build slists, forms etc. */
extern struct slist_wc *easysrc_code; /* Setopt calls etc. */
extern struct slist_wc *easysrc_toohard; /* Unconvertible setopt */
extern struct slist_wc *easysrc_clean;  /* Clean up (reverse order) */

extern int easysrc_mime_count;  /* Number of curl_mime variables */
extern int easysrc_slist_count; /* Number of curl_slist variables */

extern CURLcode easysrc_init(void);
extern CURLcode easysrc_add(struct slist_wc **plist, const char *bupf);
extern CURLcode easysrc_addf(struct slist_wc **plist,
                             const char *fmt, ...) CURL_PRINTF(2, 3);
extern CURLcode easysrc_perform(void);
extern CURLcode easysrc_cleanup(void);

void dumpeasysrc(struct GlobalConfig *config);

#else /* CURL_DISABLE_LIBCURL_OPTION is defined */

#define easysrc_init() CURLE_OK
#define easysrc_cleanup()
#define dumpeasysrc(x)
#define easysrc_perform() CURLE_OK

#endif /* CURL_DISABLE_LIBCURL_OPTION */

#endif /* HEADER_CURL_TOOL_EASYSRC_H */
