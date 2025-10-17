/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 28, 2025.
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
/* <DESC>
 * Simple HTTP GET that stores the headers in a separate file
 * </DESC>
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <curl/curl.h>

static size_t write_data(void *ptr, size_t size, size_t nmemb, void *stream)
{
  size_t written = fwrite(ptr, size, nmemb, (FILE *)stream);
  return written;
}

int main(void)
{
  CURL *curl_handle;
  static const char *headerfilename = "head.out";
  FILE *headerfile;
  static const char *bodyfilename = "body.out";
  FILE *bodyfile;

  curl_global_init(CURL_GLOBAL_ALL);

  /* init the curl session */
  curl_handle = curl_easy_init();

  /* set URL to get */
  curl_easy_setopt(curl_handle, CURLOPT_URL, "https://example.com");

  /* no progress meter please */
  curl_easy_setopt(curl_handle, CURLOPT_NOPROGRESS, 1L);

  /* send all data to this function  */
  curl_easy_setopt(curl_handle, CURLOPT_WRITEFUNCTION, write_data);

  /* open the header file */
  headerfile = fopen(headerfilename, "wb");
  if(!headerfile) {
    curl_easy_cleanup(curl_handle);
    return -1;
  }

  /* open the body file */
  bodyfile = fopen(bodyfilename, "wb");
  if(!bodyfile) {
    curl_easy_cleanup(curl_handle);
    fclose(headerfile);
    return -1;
  }

  /* we want the headers be written to this file handle */
  curl_easy_setopt(curl_handle, CURLOPT_HEADERDATA, headerfile);

  /* we want the body be written to this file handle instead of stdout */
  curl_easy_setopt(curl_handle, CURLOPT_WRITEDATA, bodyfile);

  /* get it! */
  curl_easy_perform(curl_handle);

  /* close the header file */
  fclose(headerfile);

  /* close the body file */
  fclose(bodyfile);

  /* cleanup curl stuff */
  curl_easy_cleanup(curl_handle);

  return 0;
}
