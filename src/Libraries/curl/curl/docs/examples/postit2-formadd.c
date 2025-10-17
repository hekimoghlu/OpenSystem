/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 28, 2025.
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
 * HTTP Multipart formpost with file upload and two additional parts.
 * </DESC>
 */

/*
 * Example code that uploads a filename 'foo' to a remote script that accepts
 * "HTML form based" (as described in RFC 1738) uploads using HTTP POST.
 *
 * Warning: this example uses the deprecated form api. See "postit2.c"
 *          for a similar example using the mime api.
 *
 * The imaginary form we fill in looks like:
 *
 * <form method="post" enctype="multipart/form-data" action="examplepost.cgi">
 * Enter file: <input type="file" name="sendfile" size="40">
 * Enter filename: <input type="text" name="filename" size="30">
 * <input type="submit" value="send" name="submit">
 * </form>
 */

#include <stdio.h>
#include <string.h>

#include <curl/curl.h>

int main(int argc, char *argv[])
{
  CURL *curl;
  CURLcode res;

  struct curl_httppost *formpost = NULL;
  struct curl_httppost *lastptr = NULL;
  struct curl_slist *headerlist = NULL;
  static const char buf[] = "Expect:";

  curl_global_init(CURL_GLOBAL_ALL);

  /* Fill in the file upload field */
  curl_formadd(&formpost,
               &lastptr,
               CURLFORM_COPYNAME, "sendfile",
               CURLFORM_FILE, "postit2-formadd.c",
               CURLFORM_END);

  /* Fill in the filename field */
  curl_formadd(&formpost,
               &lastptr,
               CURLFORM_COPYNAME, "filename",
               CURLFORM_COPYCONTENTS, "postit2-formadd.c",
               CURLFORM_END);


  /* Fill in the submit field too, even if this is rarely needed */
  curl_formadd(&formpost,
               &lastptr,
               CURLFORM_COPYNAME, "submit",
               CURLFORM_COPYCONTENTS, "send",
               CURLFORM_END);

  curl = curl_easy_init();
  /* initialize custom header list (stating that Expect: 100-continue is not
     wanted */
  headerlist = curl_slist_append(headerlist, buf);
  if(curl) {
    /* what URL that receives this POST */
    curl_easy_setopt(curl, CURLOPT_URL, "https://example.com/examplepost.cgi");
    if((argc == 2) && (!strcmp(argv[1], "noexpectheader")))
      /* only disable 100-continue header if explicitly requested */
      curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headerlist);
    curl_easy_setopt(curl, CURLOPT_HTTPPOST, formpost);

    /* Perform the request, res gets the return code */
    res = curl_easy_perform(curl);
    /* Check for errors */
    if(res != CURLE_OK)
      fprintf(stderr, "curl_easy_perform() failed: %s\n",
              curl_easy_strerror(res));

    /* always cleanup */
    curl_easy_cleanup(curl);

    /* then cleanup the formpost chain */
    curl_formfree(formpost);
    /* free slist */
    curl_slist_free_all(headerlist);
  }
  return 0;
}

