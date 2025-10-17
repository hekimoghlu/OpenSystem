/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 24, 2022.
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
#include "curl/urlapi.h"
#include "urlapi-int.h"


static CURLU *u;

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

#define free_and_clear(x) free(x); x = NULL

static CURLUcode parse_port(CURLU *url,
                           char *h, bool has_scheme)
{
  struct dynbuf host;
  CURLUcode ret;
  Curl_dyn_init(&host, 10000);
  if(Curl_dyn_add(&host, h))
    return CURLUE_OUT_OF_MEMORY;
  ret = Curl_parse_port(url, &host, has_scheme);
  Curl_dyn_free(&host);
  return ret;
}

UNITTEST_START
{
  CURLUcode ret;
  char *ipv6port = NULL;
  char *portnum;

  /* Valid IPv6 */
  u = curl_url();
  if(!u)
    goto fail;
  ipv6port = strdup("[fe80::250:56ff:fea7:da15]");
  if(!ipv6port)
    goto fail;
  ret = parse_port(u, ipv6port, FALSE);
  fail_unless(ret == CURLUE_OK, "parse_port returned error");
  ret = curl_url_get(u, CURLUPART_PORT, &portnum, CURLU_NO_DEFAULT_PORT);
  fail_unless(ret != CURLUE_OK, "curl_url_get portnum returned something");
  free_and_clear(ipv6port);
  curl_url_cleanup(u);

  /* Invalid IPv6 */
  u = curl_url();
  if(!u)
    goto fail;
  ipv6port = strdup("[fe80::250:56ff:fea7:da15|");
  if(!ipv6port)
    goto fail;
  ret = parse_port(u, ipv6port, FALSE);
  fail_unless(ret != CURLUE_OK, "parse_port true on error");
  free_and_clear(ipv6port);
  curl_url_cleanup(u);

  u = curl_url();
  if(!u)
    goto fail;
  ipv6port = strdup("[fe80::250:56ff;fea7:da15]:808");
  if(!ipv6port)
    goto fail;
  ret = parse_port(u, ipv6port, FALSE);
  fail_unless(ret == CURLUE_OK, "parse_port returned error");
  ret = curl_url_get(u, CURLUPART_PORT, &portnum, 0);
  fail_unless(ret == CURLUE_OK, "curl_url_get portnum returned error");
  fail_unless(portnum && !strcmp(portnum, "808"), "Check portnumber");

  curl_free(portnum);
  free_and_clear(ipv6port);
  curl_url_cleanup(u);

  /* Valid IPv6 with zone index and port number */
  u = curl_url();
  if(!u)
    goto fail;
  ipv6port = strdup("[fe80::250:56ff:fea7:da15%25eth3]:80");
  if(!ipv6port)
    goto fail;
  ret = parse_port(u, ipv6port, FALSE);
  fail_unless(ret == CURLUE_OK, "parse_port returned error");
  ret = curl_url_get(u, CURLUPART_PORT, &portnum, 0);
  fail_unless(ret == CURLUE_OK, "curl_url_get portnum returned error");
  fail_unless(portnum && !strcmp(portnum, "80"), "Check portnumber");
  curl_free(portnum);
  free_and_clear(ipv6port);
  curl_url_cleanup(u);

  /* Valid IPv6 with zone index without port number */
  u = curl_url();
  if(!u)
    goto fail;
  ipv6port = strdup("[fe80::250:56ff:fea7:da15%25eth3]");
  if(!ipv6port)
    goto fail;
  ret = parse_port(u, ipv6port, FALSE);
  fail_unless(ret == CURLUE_OK, "parse_port returned error");
  free_and_clear(ipv6port);
  curl_url_cleanup(u);

  /* Valid IPv6 with port number */
  u = curl_url();
  if(!u)
    goto fail;
  ipv6port = strdup("[fe80::250:56ff:fea7:da15]:81");
  if(!ipv6port)
    goto fail;
  ret = parse_port(u, ipv6port, FALSE);
  fail_unless(ret == CURLUE_OK, "parse_port returned error");
  ret = curl_url_get(u, CURLUPART_PORT, &portnum, 0);
  fail_unless(ret == CURLUE_OK, "curl_url_get portnum returned error");
  fail_unless(portnum && !strcmp(portnum, "81"), "Check portnumber");
  curl_free(portnum);
  free_and_clear(ipv6port);
  curl_url_cleanup(u);

  /* Valid IPv6 with syntax error in the port number */
  u = curl_url();
  if(!u)
    goto fail;
  ipv6port = strdup("[fe80::250:56ff:fea7:da15];81");
  if(!ipv6port)
    goto fail;
  ret = parse_port(u, ipv6port, FALSE);
  fail_unless(ret != CURLUE_OK, "parse_port true on error");
  free_and_clear(ipv6port);
  curl_url_cleanup(u);

  u = curl_url();
  if(!u)
    goto fail;
  ipv6port = strdup("[fe80::250:56ff:fea7:da15]80");
  if(!ipv6port)
    goto fail;
  ret = parse_port(u, ipv6port, FALSE);
  fail_unless(ret != CURLUE_OK, "parse_port true on error");
  free_and_clear(ipv6port);
  curl_url_cleanup(u);

  /* Valid IPv6 with no port after the colon, should use default if a scheme
     was used in the URL */
  u = curl_url();
  if(!u)
    goto fail;
  ipv6port = strdup("[fe80::250:56ff:fea7:da15]:");
  if(!ipv6port)
    goto fail;
  ret = parse_port(u, ipv6port, TRUE);
  fail_unless(ret == CURLUE_OK, "parse_port returned error");
  free_and_clear(ipv6port);
  curl_url_cleanup(u);

  /* Incorrect zone index syntax, but the port extractor doesn't care */
  u = curl_url();
  if(!u)
    goto fail;
  ipv6port = strdup("[fe80::250:56ff:fea7:da15!25eth3]:180");
  if(!ipv6port)
    goto fail;
  ret = parse_port(u, ipv6port, FALSE);
  fail_unless(ret == CURLUE_OK, "parse_port returned error");
  ret = curl_url_get(u, CURLUPART_PORT, &portnum, 0);
  fail_unless(ret == CURLUE_OK, "curl_url_get portnum returned error");
  fail_unless(portnum && !strcmp(portnum, "180"), "Check portnumber");
  curl_free(portnum);
  free_and_clear(ipv6port);
  curl_url_cleanup(u);

  /* Non percent-encoded zone index */
  u = curl_url();
  if(!u)
    goto fail;
  ipv6port = strdup("[fe80::250:56ff:fea7:da15%eth3]:80");
  if(!ipv6port)
    goto fail;
  ret = parse_port(u, ipv6port, FALSE);
  fail_unless(ret == CURLUE_OK, "parse_port returned error");
  free_and_clear(ipv6port);
  curl_url_cleanup(u);

  /* No scheme and no digits following the colon - not accepted. Because that
     makes (a*50):// that looks like a scheme be an acceptable input. */
  u = curl_url();
  if(!u)
    goto fail;
  ipv6port = strdup("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                    "aaaaaaaaaaaaaaaaaaaaaa:");
  if(!ipv6port)
    goto fail;
  ret = parse_port(u, ipv6port, FALSE);
  fail_unless(ret == CURLUE_BAD_PORT_NUMBER, "parse_port did wrong");
fail:
  free(ipv6port);
  curl_url_cleanup(u);

}
UNITTEST_STOP
