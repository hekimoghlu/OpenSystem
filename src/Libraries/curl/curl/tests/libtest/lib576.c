/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 29, 2022.
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
#include "test.h"

#include "testutil.h"
#include "memdebug.h"

struct chunk_data {
  int remains;
  int print_content;
};

static
long chunk_bgn(const struct curl_fileinfo *finfo, void *ptr, int remains)
{
  struct chunk_data *ch_d = ptr;
  ch_d->remains = remains;

  printf("=============================================================\n");
  printf("Remains:      %d\n", remains);
  printf("Filename:     %s\n", finfo->filename);
  if(finfo->strings.perm) {
    printf("Permissions:  %s", finfo->strings.perm);
    if(finfo->flags & CURLFINFOFLAG_KNOWN_PERM)
      printf(" (parsed => %o)", finfo->perm);
    printf("\n");
  }
  printf("Size:         %ldB\n", (long)finfo->size);
  if(finfo->strings.user)
    printf("User:         %s\n", finfo->strings.user);
  if(finfo->strings.group)
    printf("Group:        %s\n", finfo->strings.group);
  if(finfo->strings.time)
    printf("Time:         %s\n", finfo->strings.time);
  printf("Filetype:     ");
  switch(finfo->filetype) {
  case CURLFILETYPE_FILE:
    printf("regular file\n");
    break;
  case CURLFILETYPE_DIRECTORY:
    printf("directory\n");
    break;
  case CURLFILETYPE_SYMLINK:
    printf("symlink\n");
    printf("Target:       %s\n", finfo->strings.target);
    break;
  default:
    printf("other type\n");
    break;
  }
  if(finfo->filetype == CURLFILETYPE_FILE) {
    ch_d->print_content = 1;
    printf("Content:\n"
      "-------------------------------------------------------------\n");
  }
  if(strcmp(finfo->filename, "someothertext.txt") == 0) {
    printf("# THIS CONTENT WAS SKIPPED IN CHUNK_BGN CALLBACK #\n");
    return CURL_CHUNK_BGN_FUNC_SKIP;
  }
  return CURL_CHUNK_BGN_FUNC_OK;
}

static
long chunk_end(void *ptr)
{
  struct chunk_data *ch_d = ptr;
  if(ch_d->print_content) {
    ch_d->print_content = 0;
    printf("-------------------------------------------------------------\n");
  }
  if(ch_d->remains == 1)
    printf("=============================================================\n");
  return CURL_CHUNK_END_FUNC_OK;
}

int test(char *URL)
{
  CURL *handle = NULL;
  CURLcode res = CURLE_OK;
  struct chunk_data chunk_data = {0, 0};
  curl_global_init(CURL_GLOBAL_ALL);
  handle = curl_easy_init();
  if(!handle) {
    res = CURLE_OUT_OF_MEMORY;
    goto test_cleanup;
  }

  test_setopt(handle, CURLOPT_URL, URL);
  test_setopt(handle, CURLOPT_WILDCARDMATCH, 1L);
  test_setopt(handle, CURLOPT_CHUNK_BGN_FUNCTION, chunk_bgn);
  test_setopt(handle, CURLOPT_CHUNK_END_FUNCTION, chunk_end);
  test_setopt(handle, CURLOPT_CHUNK_DATA, &chunk_data);

  res = curl_easy_perform(handle);

test_cleanup:
  if(handle)
    curl_easy_cleanup(handle);
  curl_global_cleanup();
  return res;
}
