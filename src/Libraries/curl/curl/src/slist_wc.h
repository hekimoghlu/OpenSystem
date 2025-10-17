/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 11, 2023.
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

#ifndef HEADER_CURL_SLIST_WC_H
#define HEADER_CURL_SLIST_WC_H
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

/* linked-list structure with last node cache for easysrc */
struct slist_wc {
  struct curl_slist *first;
  struct curl_slist *last;
};

/*
 * NAME curl_slist_wc_append()
 *
 * DESCRIPTION
 *
 * Appends a string to a linked list. If no list exists, it will be created
 * first. Returns the new list, after appending.
 */
struct slist_wc *slist_wc_append(struct slist_wc *, const char *);

/*
 * NAME curl_slist_free_all()
 *
 * DESCRIPTION
 *
 * free a previously built curl_slist_wc.
 */
void slist_wc_free_all(struct slist_wc *);

#endif /* CURL_DISABLE_LIBCURL_OPTION */

#endif /* HEADER_CURL_SLIST_WC_H */
