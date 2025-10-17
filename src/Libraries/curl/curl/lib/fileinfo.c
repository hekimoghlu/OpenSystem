/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 16, 2021.
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
#include "curl_setup.h"
#ifndef CURL_DISABLE_FTP
#include "strdup.h"
#include "fileinfo.h"
#include "curl_memory.h"
/* The last #include file should be: */
#include "memdebug.h"

struct fileinfo *Curl_fileinfo_alloc(void)
{
  return calloc(1, sizeof(struct fileinfo));
}

void Curl_fileinfo_cleanup(struct fileinfo *finfo)
{
  if(!finfo)
    return;

  Curl_dyn_free(&finfo->buf);
  free(finfo);
}
#endif
