/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 6, 2025.
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
 * Uses the "Streaming HTML parser" to extract the href pieces in a streaming
 * manner from a downloaded HTML.
 * </DESC>
 */
/*
 * The HTML parser is found at https://github.com/arjunc77/htmlstreamparser
 */

#include <stdio.h>
#include <curl/curl.h>
#include <htmlstreamparser.h>


static size_t write_callback(void *buffer, size_t size, size_t nmemb,
                             void *hsp)
{
  size_t realsize = size * nmemb, p;
  for(p = 0; p < realsize; p++) {
    html_parser_char_parse(hsp, ((char *)buffer)[p]);
    if(html_parser_cmp_tag(hsp, "a", 1))
      if(html_parser_cmp_attr(hsp, "href", 4))
        if(html_parser_is_in(hsp, HTML_VALUE_ENDED)) {
          html_parser_val(hsp)[html_parser_val_length(hsp)] = '\0';
          printf("%s\n", html_parser_val(hsp));
        }
  }
  return realsize;
}

int main(int argc, char *argv[])
{
  char tag[1], attr[4], val[128];
  CURL *curl;
  HTMLSTREAMPARSER *hsp;

  if(argc != 2) {
    printf("Usage: %s URL\n", argv[0]);
    return EXIT_FAILURE;
  }

  curl = curl_easy_init();

  hsp = html_parser_init();

  html_parser_set_tag_to_lower(hsp, 1);
  html_parser_set_attr_to_lower(hsp, 1);
  html_parser_set_tag_buffer(hsp, tag, sizeof(tag));
  html_parser_set_attr_buffer(hsp, attr, sizeof(attr));
  html_parser_set_val_buffer(hsp, val, sizeof(val)-1);

  curl_easy_setopt(curl, CURLOPT_URL, argv[1]);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, hsp);
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

  curl_easy_perform(curl);

  curl_easy_cleanup(curl);

  html_parser_cleanup(hsp);

  return EXIT_SUCCESS;
}
