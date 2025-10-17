/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 19, 2025.
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

#include <curl/curl.h>

#include "llist.h"
#include "curl_memory.h"

/* this must be the last include file */
#include "memdebug.h"

/*
 * @unittest: 1300
 */
void
Curl_llist_init(struct Curl_llist *l, Curl_llist_dtor dtor)
{
  l->size = 0;
  l->dtor = dtor;
  l->head = NULL;
  l->tail = NULL;
}

/*
 * Curl_llist_insert_next()
 *
 * Inserts a new list element after the given one 'e'. If the given existing
 * entry is NULL and the list already has elements, the new one will be
 * inserted first in the list.
 *
 * The 'ne' argument should be a pointer into the object to store.
 *
 * @unittest: 1300
 */
void
Curl_llist_insert_next(struct Curl_llist *list, struct Curl_llist_element *e,
                       const void *p,
                       struct Curl_llist_element *ne)
{
  ne->ptr = (void *) p;
  if(list->size == 0) {
    list->head = ne;
    list->head->prev = NULL;
    list->head->next = NULL;
    list->tail = ne;
  }
  else {
    /* if 'e' is NULL here, we insert the new element first in the list */
    ne->next = e?e->next:list->head;
    ne->prev = e;
    if(!e) {
      list->head->prev = ne;
      list->head = ne;
    }
    else if(e->next) {
      e->next->prev = ne;
    }
    else {
      list->tail = ne;
    }
    if(e)
      e->next = ne;
  }

  ++list->size;
}

/*
 * @unittest: 1300
 */
void
Curl_llist_remove(struct Curl_llist *list, struct Curl_llist_element *e,
                  void *user)
{
  void *ptr;
  if(!e || list->size == 0)
    return;

  if(e == list->head) {
    list->head = e->next;

    if(!list->head)
      list->tail = NULL;
    else
      e->next->prev = NULL;
  }
  else {
    if(e->prev)
      e->prev->next = e->next;

    if(!e->next)
      list->tail = e->prev;
    else
      e->next->prev = e->prev;
  }

  ptr = e->ptr;

  e->ptr  = NULL;
  e->prev = NULL;
  e->next = NULL;

  --list->size;

  /* call the dtor() last for when it actually frees the 'e' memory itself */
  if(list->dtor)
    list->dtor(user, ptr);
}

void
Curl_llist_destroy(struct Curl_llist *list, void *user)
{
  if(list) {
    while(list->size > 0)
      Curl_llist_remove(list, list->tail, user);
  }
}

size_t
Curl_llist_count(struct Curl_llist *list)
{
  return list->size;
}
