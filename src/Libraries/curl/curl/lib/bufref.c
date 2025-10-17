/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 4, 2024.
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
#include "curl_setup.h"
#include "urldata.h"
#include "bufref.h"
#include "strdup.h"

#include "curl_memory.h"
#include "memdebug.h"

#define SIGNATURE 0x5c48e9b2    /* Random pattern. */

/*
 * Init a bufref struct.
 */
void Curl_bufref_init(struct bufref *br)
{
  DEBUGASSERT(br);
  br->dtor = NULL;
  br->ptr = NULL;
  br->len = 0;

#ifdef DEBUGBUILD
  br->signature = SIGNATURE;
#endif
}

/*
 * Free the buffer and re-init the necessary fields. It doesn't touch the
 * 'signature' field and thus this buffer reference can be reused.
 */

void Curl_bufref_free(struct bufref *br)
{
  DEBUGASSERT(br);
  DEBUGASSERT(br->signature == SIGNATURE);
  DEBUGASSERT(br->ptr || !br->len);

  if(br->ptr && br->dtor)
    br->dtor((void *) br->ptr);

  br->dtor = NULL;
  br->ptr = NULL;
  br->len = 0;
}

/*
 * Set the buffer reference to new values. The previously referenced buffer
 * is released before assignment.
 */
void Curl_bufref_set(struct bufref *br, const void *ptr, size_t len,
                     void (*dtor)(void *))
{
  DEBUGASSERT(ptr || !len);
  DEBUGASSERT(len <= CURL_MAX_INPUT_LENGTH);

  Curl_bufref_free(br);
  br->ptr = (const unsigned char *) ptr;
  br->len = len;
  br->dtor = dtor;
}

/*
 * Get a pointer to the referenced buffer.
 */
const unsigned char *Curl_bufref_ptr(const struct bufref *br)
{
  DEBUGASSERT(br);
  DEBUGASSERT(br->signature == SIGNATURE);
  DEBUGASSERT(br->ptr || !br->len);

  return br->ptr;
}

/*
 * Get the length of the referenced buffer data.
 */
size_t Curl_bufref_len(const struct bufref *br)
{
  DEBUGASSERT(br);
  DEBUGASSERT(br->signature == SIGNATURE);
  DEBUGASSERT(br->ptr || !br->len);

  return br->len;
}

CURLcode Curl_bufref_memdup(struct bufref *br, const void *ptr, size_t len)
{
  unsigned char *cpy = NULL;

  DEBUGASSERT(br);
  DEBUGASSERT(br->signature == SIGNATURE);
  DEBUGASSERT(br->ptr || !br->len);
  DEBUGASSERT(ptr || !len);
  DEBUGASSERT(len <= CURL_MAX_INPUT_LENGTH);

  if(ptr) {
    cpy = Curl_memdup0(ptr, len);
    if(!cpy)
      return CURLE_OUT_OF_MEMORY;
  }

  Curl_bufref_set(br, cpy, len, curl_free);
  return CURLE_OK;
}
