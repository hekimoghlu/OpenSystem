/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 14, 2022.
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
#include "tool_setup.h"

#ifndef CURL_DISABLE_LIBCURL_OPTION

#include "slist_wc.h"

/* The last #include files should be: */
#include "memdebug.h"

/*
 * slist_wc_append() appends a string to the linked list. This function can be
 * used as an initialization function as well as an append function.
 */
struct slist_wc *slist_wc_append(struct slist_wc *list,
                                 const char *data)
{
  struct curl_slist *new_item = curl_slist_append(NULL, data);

  if(!new_item)
    return NULL;

  if(!list) {
    list = malloc(sizeof(struct slist_wc));

    if(!list) {
      curl_slist_free_all(new_item);
      return NULL;
    }

    list->first = new_item;
    list->last = new_item;
    return list;
  }

  list->last->next = new_item;
  list->last = list->last->next;
  return list;
}

/* be nice and clean up resources */
void slist_wc_free_all(struct slist_wc *list)
{
  if(!list)
    return;

  curl_slist_free_all(list->first);
  free(list);
}

#endif /* CURL_DISABLE_LIBCURL_OPTION */
