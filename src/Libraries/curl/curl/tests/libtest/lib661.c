/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 26, 2023.
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
#include "memdebug.h"

int test(char *URL)
{
   CURLcode res;
   CURL *curl = NULL;
   char *newURL = NULL;
   struct curl_slist *slist = NULL;

   if(curl_global_init(CURL_GLOBAL_ALL) != CURLE_OK) {
     fprintf(stderr, "curl_global_init() failed\n");
     return TEST_ERR_MAJOR_BAD;
   }

   curl = curl_easy_init();
   if(!curl) {
     fprintf(stderr, "curl_easy_init() failed\n");
     res = TEST_ERR_MAJOR_BAD;
     goto test_cleanup;
   }

   /* test: CURLFTPMETHOD_SINGLECWD with absolute path should
            skip CWD to entry path */
   newURL = aprintf("%s/folderA/661", URL);
   test_setopt(curl, CURLOPT_URL, newURL);
   test_setopt(curl, CURLOPT_VERBOSE, 1L);
   test_setopt(curl, CURLOPT_IGNORE_CONTENT_LENGTH, 1L);
   test_setopt(curl, CURLOPT_FTP_FILEMETHOD, (long) CURLFTPMETHOD_SINGLECWD);
   res = curl_easy_perform(curl);
   if(res != CURLE_REMOTE_FILE_NOT_FOUND)
     goto test_cleanup;

   curl_free(newURL);
   newURL = aprintf("%s/folderB/661", URL);
   test_setopt(curl, CURLOPT_URL, newURL);
   res = curl_easy_perform(curl);
   if(res != CURLE_REMOTE_FILE_NOT_FOUND)
     goto test_cleanup;

   /* test: CURLFTPMETHOD_NOCWD with absolute path should
      never emit CWD (for both new and reused easy handle) */
   curl_easy_cleanup(curl);
   curl = curl_easy_init();
   if(!curl) {
     fprintf(stderr, "curl_easy_init() failed\n");
     res = TEST_ERR_MAJOR_BAD;
     goto test_cleanup;
   }

   curl_free(newURL);
   newURL = aprintf("%s/folderA/661", URL);
   test_setopt(curl, CURLOPT_URL, newURL);
   test_setopt(curl, CURLOPT_VERBOSE, 1L);
   test_setopt(curl, CURLOPT_IGNORE_CONTENT_LENGTH, 1L);
   test_setopt(curl, CURLOPT_FTP_FILEMETHOD, (long) CURLFTPMETHOD_NOCWD);
   res = curl_easy_perform(curl);
   if(res != CURLE_REMOTE_FILE_NOT_FOUND)
     goto test_cleanup;

   /* curve ball: CWD /folderB before reusing connection with _NOCWD */
   curl_free(newURL);
   newURL = aprintf("%s/folderB/661", URL);
   test_setopt(curl, CURLOPT_URL, newURL);
   test_setopt(curl, CURLOPT_FTP_FILEMETHOD, (long) CURLFTPMETHOD_SINGLECWD);
   res = curl_easy_perform(curl);
   if(res != CURLE_REMOTE_FILE_NOT_FOUND)
     goto test_cleanup;

   curl_free(newURL);
   newURL = aprintf("%s/folderA/661", URL);
   test_setopt(curl, CURLOPT_URL, newURL);
   test_setopt(curl, CURLOPT_FTP_FILEMETHOD, (long) CURLFTPMETHOD_NOCWD);
   res = curl_easy_perform(curl);
   if(res != CURLE_REMOTE_FILE_NOT_FOUND)
     goto test_cleanup;

   /* test: CURLFTPMETHOD_NOCWD with home-relative path should
      not emit CWD for first FTP access after login */
   curl_easy_cleanup(curl);
   curl = curl_easy_init();
   if(!curl) {
     fprintf(stderr, "curl_easy_init() failed\n");
     res = TEST_ERR_MAJOR_BAD;
     goto test_cleanup;
   }

   slist = curl_slist_append(NULL, "SYST");
   if(!slist) {
     fprintf(stderr, "curl_slist_append() failed\n");
     res = TEST_ERR_MAJOR_BAD;
     goto test_cleanup;
   }

   test_setopt(curl, CURLOPT_URL, URL);
   test_setopt(curl, CURLOPT_VERBOSE, 1L);
   test_setopt(curl, CURLOPT_NOBODY, 1L);
   test_setopt(curl, CURLOPT_FTP_FILEMETHOD, (long) CURLFTPMETHOD_NOCWD);
   test_setopt(curl, CURLOPT_QUOTE, slist);
   res = curl_easy_perform(curl);
   if(res)
     goto test_cleanup;

   /* test: CURLFTPMETHOD_SINGLECWD with home-relative path should
      not emit CWD for first FTP access after login */
   curl_easy_cleanup(curl);
   curl = curl_easy_init();
   if(!curl) {
     fprintf(stderr, "curl_easy_init() failed\n");
     res = TEST_ERR_MAJOR_BAD;
     goto test_cleanup;
   }

   test_setopt(curl, CURLOPT_URL, URL);
   test_setopt(curl, CURLOPT_VERBOSE, 1L);
   test_setopt(curl, CURLOPT_NOBODY, 1L);
   test_setopt(curl, CURLOPT_FTP_FILEMETHOD, (long) CURLFTPMETHOD_SINGLECWD);
   test_setopt(curl, CURLOPT_QUOTE, slist);
   res = curl_easy_perform(curl);
   if(res)
     goto test_cleanup;

   /* test: CURLFTPMETHOD_NOCWD with home-relative path should
      not emit CWD for second FTP access when not needed +
      bonus: see if path buffering survives curl_easy_reset() */
   curl_easy_reset(curl);
   test_setopt(curl, CURLOPT_URL, URL);
   test_setopt(curl, CURLOPT_VERBOSE, 1L);
   test_setopt(curl, CURLOPT_NOBODY, 1L);
   test_setopt(curl, CURLOPT_FTP_FILEMETHOD, (long) CURLFTPMETHOD_NOCWD);
   test_setopt(curl, CURLOPT_QUOTE, slist);
   res = curl_easy_perform(curl);


test_cleanup:

   if(res)
     fprintf(stderr, "test encountered error %d\n", res);
   curl_slist_free_all(slist);
   curl_free(newURL);
   curl_easy_cleanup(curl);
   curl_global_cleanup();

   return (int)res;
}

