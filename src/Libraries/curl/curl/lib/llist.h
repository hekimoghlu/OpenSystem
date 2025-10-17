/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 22, 2023.
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

#ifndef HEADER_CURL_LLIST_H
#define HEADER_CURL_LLIST_H
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

#include "curl_setup.h"
#include <stddef.h>

typedef void (*Curl_llist_dtor)(void *, void *);

struct Curl_llist_element {
  void *ptr;
  struct Curl_llist_element *prev;
  struct Curl_llist_element *next;
};

struct Curl_llist {
  struct Curl_llist_element *head;
  struct Curl_llist_element *tail;
  Curl_llist_dtor dtor;
  size_t size;
};

void Curl_llist_init(struct Curl_llist *, Curl_llist_dtor);
void Curl_llist_insert_next(struct Curl_llist *, struct Curl_llist_element *,
                            const void *, struct Curl_llist_element *node);
void Curl_llist_remove(struct Curl_llist *, struct Curl_llist_element *,
                       void *);
size_t Curl_llist_count(struct Curl_llist *);
void Curl_llist_destroy(struct Curl_llist *, void *);
#endif /* HEADER_CURL_LLIST_H */
