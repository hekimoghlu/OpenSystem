/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 20, 2023.
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

#ifndef HEADER_CURL_TOOL_CB_HDR_H
#define HEADER_CURL_TOOL_CB_HDR_H
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

/*
 * curl operates using a single HdrCbData struct variable, a
 * pointer to this is passed as userdata pointer to tool_header_cb.
 *
 * 'outs' member is a pointer to the OutStruct variable used to keep
 * track of information relative to curl's output writing.
 *
 * 'heads' member is a pointer to the OutStruct variable used to keep
 * track of information relative to header response writing.
 *
 * 'honor_cd_filename' member is TRUE when tool_header_cb is allowed
 * to honor Content-Disposition filename property and accordingly
 * set 'outs' filename, otherwise FALSE;
 */

struct HdrCbData {
  struct GlobalConfig *global;
  struct OperationConfig *config;
  struct OutStruct *outs;
  struct OutStruct *heads;
  struct OutStruct *etag_save;
  bool honor_cd_filename;
};

/*
** callback for CURLOPT_HEADERFUNCTION
*/

size_t tool_header_cb(char *ptr, size_t size, size_t nmemb, void *userdata);

#endif /* HEADER_CURL_TOOL_CB_HDR_H */
