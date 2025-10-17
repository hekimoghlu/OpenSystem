/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 7, 2022.
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
#include <stdio.h>
#include <string.h>

#include <curl/curl.h>

/* <DESC>
 * Checks a single file's size and mtime from an FTP server.
 * </DESC>
 */

static size_t throw_away(void *ptr, size_t size, size_t nmemb, void *data)
{
  (void)ptr;
  (void)data;
  /* we are not interested in the headers itself,
     so we only return the size we would have saved ... */
  return (size_t)(size * nmemb);
}

int main(void)
{
  char ftpurl[] = "ftp://ftp.example.com/gnu/binutils/binutils-2.19.1.tar.bz2";
  CURL *curl;
  CURLcode res;
  long filetime = -1;
  curl_off_t filesize = 0;
  const char *filename = strrchr(ftpurl, '/') + 1;

  curl_global_init(CURL_GLOBAL_DEFAULT);

  curl = curl_easy_init();
  if(curl) {
    curl_easy_setopt(curl, CURLOPT_URL, ftpurl);
    /* No download if the file */
    curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);
    /* Ask for filetime */
    curl_easy_setopt(curl, CURLOPT_FILETIME, 1L);
    curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, throw_away);
    curl_easy_setopt(curl, CURLOPT_HEADER, 0L);
    /* Switch on full protocol/debug output */
    /* curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L); */

    res = curl_easy_perform(curl);

    if(CURLE_OK == res) {
      /* https://curl.se/libcurl/c/curl_easy_getinfo.html */
      res = curl_easy_getinfo(curl, CURLINFO_FILETIME, &filetime);
      if((CURLE_OK == res) && (filetime >= 0)) {
        time_t file_time = (time_t)filetime;
        printf("filetime %s: %s", filename, ctime(&file_time));
      }
      res = curl_easy_getinfo(curl, CURLINFO_CONTENT_LENGTH_DOWNLOAD_T,
                              &filesize);
      if((CURLE_OK == res) && (filesize>0))
        printf("filesize %s: %" CURL_FORMAT_CURL_OFF_T " bytes\n",
               filename, filesize);
    }
    else {
      /* we failed */
      fprintf(stderr, "curl told us %d\n", res);
    }

    /* always cleanup */
    curl_easy_cleanup(curl);
  }

  curl_global_cleanup();

  return 0;
}
