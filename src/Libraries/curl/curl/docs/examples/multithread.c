/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 7, 2024.
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
 * A multi-threaded program using pthreads to fetch several files at once
 * </DESC>
 */

#include <stdio.h>
#include <pthread.h>
#include <curl/curl.h>

#define NUMT 4

/*
  List of URLs to fetch.

  If you intend to use a SSL-based protocol here you might need to setup TLS
  library mutex callbacks as described here:

  https://curl.se/libcurl/c/threadsafe.html

*/
const char * const urls[NUMT]= {
  "https://curl.se/",
  "ftp://example.com/",
  "https://example.net/",
  "www.example"
};

static void *pull_one_url(void *url)
{
  CURL *curl;

  curl = curl_easy_init();
  curl_easy_setopt(curl, CURLOPT_URL, url);
  curl_easy_perform(curl); /* ignores error */
  curl_easy_cleanup(curl);

  return NULL;
}


/*
   int pthread_create(pthread_t *new_thread_ID,
   const pthread_attr_t *attr,
   void * (*start_func)(void *), void *arg);
*/

int main(int argc, char **argv)
{
  pthread_t tid[NUMT];
  int i;

  /* Must initialize libcurl before any threads are started */
  curl_global_init(CURL_GLOBAL_ALL);

  for(i = 0; i< NUMT; i++) {
    int error = pthread_create(&tid[i],
                               NULL, /* default attributes please */
                               pull_one_url,
                               (void *)urls[i]);
    if(0 != error)
      fprintf(stderr, "Couldn't run thread number %d, errno %d\n", i, error);
    else
      fprintf(stderr, "Thread %d, gets %s\n", i, urls[i]);
  }

  /* now wait for all threads to terminate */
  for(i = 0; i< NUMT; i++) {
    pthread_join(tid[i], NULL);
    fprintf(stderr, "Thread %d terminated\n", i);
  }
  curl_global_cleanup();
  return 0;
}
