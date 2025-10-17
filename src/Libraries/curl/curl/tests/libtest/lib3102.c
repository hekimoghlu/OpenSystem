/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 11, 2023.
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

/*
 * Verify correct order of certificates in the chain by comparing the
 * subject and issuer attributes of each certificate.
 */
static bool is_chain_in_order(struct curl_certinfo *cert_info)
{
  char *last_issuer = NULL;
  int cert;

  /* Chains with only a single certificate are always in order */
  if(cert_info->num_of_certs <= 1)
    return 1;

  /* Enumerate each certificate in the chain */
  for(cert = 0; cert < cert_info->num_of_certs; cert++) {
    struct curl_slist *slist = cert_info->certinfo[cert];
    char *issuer = NULL;
    char *subject = NULL;

    /* Find the certificate issuer and subject by enumerating each field */
    for(; slist && (!issuer || !subject); slist = slist->next) {
      const char issuer_prefix[] = "Issuer:";
      const char subject_prefix[] = "Subject:";

      if(!strncmp(slist->data, issuer_prefix, sizeof(issuer_prefix)-1)) {
        issuer = slist->data + sizeof(issuer_prefix)-1;
      }
      if(!strncmp(slist->data, subject_prefix, sizeof(subject_prefix)-1)) {
        subject = slist->data + sizeof(subject_prefix)-1;
      }
    }

    if(subject && issuer) {
      printf("cert %d\n", cert);
      printf("  subject: %s\n", subject);
      printf("  issuer: %s\n", issuer);

      if(last_issuer) {
        /* If the last certificate's issuer matches the current certificate's
         * subject, then the chain is in order */
        if(strcmp(last_issuer, subject) != 0) {
          fprintf(stderr, "cert %d issuer does not match cert %d subject\n",
                  cert - 1, cert);
          fprintf(stderr, "certificate chain is not in order\n");
          return false;
        }
      }
    }

    last_issuer = issuer;
  }

  printf("certificate chain is in order\n");
  return true;
}

static size_t wrfu(void *ptr,  size_t  size,  size_t  nmemb,  void *stream)
{
  (void)stream;
  (void)ptr;
  return size * nmemb;
}

int test(char *URL)
{
  CURL *curl;
  CURLcode res = CURLE_OK;

  if(curl_global_init(CURL_GLOBAL_ALL) != CURLE_OK) {
    fprintf(stderr, "curl_global_init() failed\n");
    return TEST_ERR_MAJOR_BAD;
  }

  curl = curl_easy_init();
  if(!curl) {
    fprintf(stderr, "curl_easy_init() failed\n");
    curl_global_cleanup();
    return TEST_ERR_MAJOR_BAD;
  }

  /* Set the HTTPS url to retrieve. */
  test_setopt(curl, CURLOPT_URL, URL);

  /* Capture certificate information */
  test_setopt(curl, CURLOPT_CERTINFO, 1L);

  /* Ignore output */
  test_setopt(curl, CURLOPT_WRITEFUNCTION, wrfu);

  /* No peer verify */
  test_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
  test_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);

  /* Perform the request, res will get the return code */
  res = curl_easy_perform(curl);
  if(!res || res == CURLE_GOT_NOTHING) {
    struct curl_certinfo *cert_info = NULL;
    /* Get the certificate information */
    res = curl_easy_getinfo(curl, CURLINFO_CERTINFO, &cert_info);
    if(!res) {
      /* Check to see if the certificate chain is ordered correctly */
      if(!is_chain_in_order(cert_info))
        res = TEST_ERR_FAILURE;
    }
  }

test_cleanup:

  /* always cleanup */
  curl_easy_cleanup(curl);
  curl_global_cleanup();

  return res;
}
