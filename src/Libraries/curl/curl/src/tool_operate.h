/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 8, 2025.
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

#ifndef HEADER_CURL_TOOL_OPERATE_H
#define HEADER_CURL_TOOL_OPERATE_H
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
#include "tool_cb_hdr.h"
#include "tool_cb_prg.h"
#include "tool_sdecls.h"

struct per_transfer {
  /* double linked */
  struct per_transfer *next;
  struct per_transfer *prev;
  struct OperationConfig *config; /* for this transfer */
  struct curl_certinfo *certinfo;
  CURL *curl;
  long retry_numretries;
  long retry_sleep_default;
  long retry_sleep;
  struct timeval start; /* start of this transfer */
  struct timeval retrystart;
  char *this_url;
  unsigned int urlnum; /* the index of the given URL */
  char *outfile;
  bool infdopen; /* TRUE if infd needs closing */
  int infd;
  bool noprogress;
  struct ProgressData progressbar;
  struct OutStruct outs;
  struct OutStruct heads;
  struct OutStruct etag_save;
  struct HdrCbData hdrcbdata;
  long num_headers;
  bool was_last_header_empty;

  bool added; /* set TRUE when added to the multi handle */
  time_t startat; /* when doing parallel transfers, this is a retry transfer
                     that has been set to sleep until this time before it
                     should get started (again) */
  bool abort; /* when doing parallel transfers and this is TRUE then a critical
                 error (eg --fail-early) has occurred in another transfer and
                 this transfer will be aborted in the progress callback */

  /* for parallel progress bar */
  curl_off_t dltotal;
  curl_off_t dlnow;
  curl_off_t ultotal;
  curl_off_t ulnow;
  curl_off_t uploadfilesize; /* expected total amount */
  curl_off_t uploadedsofar; /* amount delivered from the callback */
  bool dltotal_added; /* if the total has been added from this */
  bool ultotal_added;

  /* NULL or malloced */
  char *uploadfile;
  char *errorbuffer; /* allocated and assigned while this is used for a
                        transfer */
};

CURLcode operate(struct GlobalConfig *config, int argc, argv_item_t argv[]);
void single_transfer_cleanup(struct OperationConfig *config);

extern struct per_transfer *transfers; /* first node */

#endif /* HEADER_CURL_TOOL_OPERATE_H */
