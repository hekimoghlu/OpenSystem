/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 1, 2022.
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
#include "curlcheck.h"

#include "urldata.h"
#include "hsts.h"

static CURLcode
unit_setup(void)
{
  return CURLE_OK;
}

static void
unit_stop(void)
{
  curl_global_cleanup();
}

#if defined(CURL_DISABLE_HTTP) || defined(CURL_DISABLE_HSTS)
UNITTEST_START
{
  return 0; /* nothing to do when HTTP or HSTS are disabled */
}
UNITTEST_STOP
#else

struct testit {
  const char *host;
  const char *chost; /* if non-NULL, use to lookup with */
  const char *hdr; /* if NULL, just do the lookup */
  const CURLcode result; /* parse result */
};

static const struct testit headers[] = {
  /* two entries read from disk cache, verify first */
  { "-", "readfrom.example", NULL, CURLE_OK},
  { "-", "old.example", NULL, CURLE_OK},
  /* delete the remaining one read from disk */
  { "readfrom.example", NULL, "max-age=\"0\"", CURLE_OK},

  { "example.com", NULL, "max-age=\"31536000\"\r\n", CURLE_OK },
  { "example.com", NULL, "max-age=\"21536000\"\r\n", CURLE_OK },
  { "example.com", NULL, "max-age=\"21536000\"; \r\n", CURLE_OK },
  { "example.com", NULL, "max-age=\"21536000\"; includeSubDomains\r\n",
    CURLE_OK },
  { "example.org", NULL, "max-age=\"31536000\"\r\n", CURLE_OK },
  { "this.example", NULL, "max=\"31536\";", CURLE_BAD_FUNCTION_ARGUMENT },
  { "this.example", NULL, "max-age=\"31536", CURLE_BAD_FUNCTION_ARGUMENT },
  { "this.example", NULL, "max-age=31536\"", CURLE_OK },
  /* max-age=0 removes the entry */
  { "this.example", NULL, "max-age=0", CURLE_OK },
  { "another.example", NULL, "includeSubDomains; ",
    CURLE_BAD_FUNCTION_ARGUMENT },

  /* Two max-age is illegal */
  { "example.com", NULL,
    "max-age=\"21536000\"; includeSubDomains; max-age=\"3\";",
    CURLE_BAD_FUNCTION_ARGUMENT },
  /* Two includeSubDomains is illegal */
  { "2.example.com", NULL,
    "max-age=\"21536000\"; includeSubDomains; includeSubDomains;",
    CURLE_BAD_FUNCTION_ARGUMENT },
  /* use a unknown directive "include" that should be ignored */
  { "3.example.com", NULL, "max-age=\"21536000\"; include; includeSubDomains;",
    CURLE_OK },
  /* remove the "3.example.com" one, should still match the example.com */
  { "3.example.com", NULL, "max-age=\"0\"; includeSubDomains;",
    CURLE_OK },
  { "-", "foo.example.com", NULL, CURLE_OK},
  { "-", "foo.xample.com", NULL, CURLE_OK},

  /* should not match */
  { "example.net", "forexample.net", "max-age=\"31536000\"\r\n", CURLE_OK },

  /* should not match either, since forexample.net is not in the example.net
     domain */
  { "example.net", "forexample.net",
    "max-age=\"31536000\"; includeSubDomains\r\n", CURLE_OK },
  /* remove example.net again */
  { "example.net", NULL, "max-age=\"0\"; includeSubDomains\r\n", CURLE_OK },

  /* make this live for 7 seconds */
  { "expire.example", NULL, "max-age=\"7\"\r\n", CURLE_OK },
  { NULL, NULL, NULL, CURLE_OK }
};

static void showsts(struct stsentry *e, const char *chost)
{
  if(!e)
    printf("'%s' is not HSTS\n", chost);
  else {
    printf("%s [%s]: %" CURL_FORMAT_CURL_OFF_T "%s\n",
           chost, e->host, e->expires,
           e->includeSubDomains ? " includeSubDomains" : "");
  }
}

UNITTEST_START
  CURLcode result;
  struct stsentry *e;
  struct hsts *h = Curl_hsts_init();
  int i;
  const char *chost;
  CURL *easy;
  char savename[256];

  abort_unless(h, "Curl_hsts_init()");

  curl_global_init(CURL_GLOBAL_ALL);
  easy = curl_easy_init();
  if(!easy) {
    Curl_hsts_cleanup(&h);
    curl_global_cleanup();
    abort_unless(easy, "curl_easy_init()");
  }

  Curl_hsts_loadfile(easy, h, arg);

  for(i = 0; headers[i].host ; i++) {
    if(headers[i].hdr) {
      result = Curl_hsts_parse(h, headers[i].host, headers[i].hdr);

      if(result != headers[i].result) {
        fprintf(stderr, "Curl_hsts_parse(%s) failed: %d\n",
                headers[i].hdr, result);
        unitfail++;
        continue;
      }
      else if(result) {
        printf("Input %u: error %d\n", i, (int) result);
        continue;
      }
    }

    chost = headers[i].chost ? headers[i].chost : headers[i].host;
    e = Curl_hsts(h, chost, TRUE);
    showsts(e, chost);
  }

  printf("Number of entries: %zu\n", h->list.size);

  /* verify that it is exists for 7 seconds */
  chost = "expire.example";
  for(i = 100; i < 110; i++) {
    e = Curl_hsts(h, chost, TRUE);
    showsts(e, chost);
    deltatime++; /* another second passed */
  }

  msnprintf(savename, sizeof(savename), "%s.save", arg);
  (void)Curl_hsts_save(easy, h, savename);
  Curl_hsts_cleanup(&h);
  curl_easy_cleanup(easy);
  curl_global_cleanup();

UNITTEST_STOP
#endif
