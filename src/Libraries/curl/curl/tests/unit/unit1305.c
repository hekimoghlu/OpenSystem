/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 25, 2021.
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

#ifdef HAVE_NETINET_IN_H
#  include <netinet/in.h>
#endif
#ifdef HAVE_NETDB_H
#  include <netdb.h>
#endif
#ifdef HAVE_ARPA_INET_H
#  include <arpa/inet.h>
#endif

#define ENABLE_CURLX_PRINTF
#include "curlx.h"

#include "hash.h"
#include "hostip.h"

#include "memdebug.h" /* LAST include file */

static struct Curl_easy *data;
static struct Curl_hash hp;
static char *data_key;
static struct Curl_dns_entry *data_node;

static CURLcode unit_setup(void)
{
  data = curl_easy_init();
  if(!data) {
    curl_global_cleanup();
    return CURLE_OUT_OF_MEMORY;
  }

  Curl_init_dnscache(&hp, 7);
  return CURLE_OK;
}

static void unit_stop(void)
{
  if(data_node) {
    Curl_freeaddrinfo(data_node->addr);
    free(data_node);
  }
  free(data_key);
  Curl_hash_destroy(&hp);

  curl_easy_cleanup(data);
  curl_global_cleanup();
}

static struct Curl_addrinfo *fake_ai(void)
{
  static struct Curl_addrinfo *ai;
  static const char dummy[]="dummy";
  size_t namelen = sizeof(dummy); /* including the null-terminator */

  ai = calloc(1, sizeof(struct Curl_addrinfo) + sizeof(struct sockaddr_in) +
              namelen);
  if(!ai)
    return NULL;

  ai->ai_addr = (void *)((char *)ai + sizeof(struct Curl_addrinfo));
  ai->ai_canonname = (void *)((char *)ai->ai_addr +
                              sizeof(struct sockaddr_in));
  memcpy(ai->ai_canonname, dummy, namelen);

  ai->ai_family = AF_INET;
  ai->ai_addrlen = sizeof(struct sockaddr_in);

  return ai;
}

static CURLcode create_node(void)
{
  data_key = aprintf("%s:%d", "dummy", 0);
  if(!data_key)
    return CURLE_OUT_OF_MEMORY;

  data_node = calloc(1, sizeof(struct Curl_dns_entry));
  if(!data_node)
    return CURLE_OUT_OF_MEMORY;

  data_node->addr = fake_ai();
  if(!data_node->addr)
    return CURLE_OUT_OF_MEMORY;

  return CURLE_OK;
}


UNITTEST_START

  struct Curl_dns_entry *nodep;
  size_t key_len;

  /* Test 1305 exits without adding anything to the hash */
  if(strcmp(arg, "1305") != 0) {
    CURLcode rc = create_node();
    abort_unless(rc == CURLE_OK, "data node creation failed");
    key_len = strlen(data_key);

    data_node->inuse = 1; /* hash will hold the reference */
    nodep = Curl_hash_add(&hp, data_key, key_len + 1, data_node);
    abort_unless(nodep, "insertion into hash failed");
    /* Freeing will now be done by Curl_hash_destroy */
    data_node = NULL;
  }

UNITTEST_STOP
