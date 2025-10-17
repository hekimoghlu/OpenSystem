/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 15, 2025.
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

#include "splay.h"
#include "warnless.h"


static CURLcode unit_setup(void)
{
  return CURLE_OK;
}

static void unit_stop(void)
{

}

static void splayprint(struct Curl_tree *t, int d, char output)
{
  struct Curl_tree *node;
  int i;
  int count;
  if(!t)
    return;

  splayprint(t->larger, d + 1, output);
  for(i = 0; i<d; i++)
    if(output)
      printf("  ");

  if(output) {
    printf("%ld.%ld[%d]", (long)t->key.tv_sec,
           (long)t->key.tv_usec, i);
  }

  for(count = 0, node = t->samen; node != t; node = node->samen, count++)
    ;

  if(output) {
    if(count)
      printf(" [%d more]\n", count);
    else
      printf("\n");
  }

  splayprint(t->smaller, d + 1, output);
}

UNITTEST_START

/* number of nodes to add to the splay tree */
#define NUM_NODES 50

  struct Curl_tree *root, *removed;
  struct Curl_tree nodes[NUM_NODES*3];
  size_t storage[NUM_NODES*3];
  int rc;
  int i, j;
  struct curltime tv_now = {0, 0};
  root = NULL;              /* the empty tree */

  /* add nodes */
  for(i = 0; i < NUM_NODES; i++) {
    struct curltime key;

    key.tv_sec = 0;
    key.tv_usec = (541*i)%1023;
    storage[i] = key.tv_usec;
    nodes[i].payload = &storage[i];
    root = Curl_splayinsert(key, root, &nodes[i]);
  }

  puts("Result:");
  splayprint(root, 0, 1);

  for(i = 0; i < NUM_NODES; i++) {
    int rem = (i + 7)%NUM_NODES;
    printf("Tree look:\n");
    splayprint(root, 0, 1);
    printf("remove pointer %d, payload %zu\n", rem,
           *(size_t *)nodes[rem].payload);
    rc = Curl_splayremove(root, &nodes[rem], &root);
    if(rc) {
      /* failed! */
      printf("remove %d failed!\n", rem);
      fail("remove");
    }
  }

  fail_unless(root == NULL, "tree not empty after removing all nodes");

  /* rebuild tree */
  for(i = 0; i < NUM_NODES; i++) {
    struct curltime key;

    key.tv_sec = 0;
    key.tv_usec = (541*i)%1023;

    /* add some nodes with the same key */
    for(j = 0; j <= i % 3; j++) {
      storage[i * 3 + j] = key.tv_usec*10 + j;
      nodes[i * 3 + j].payload = &storage[i * 3 + j];
      root = Curl_splayinsert(key, root, &nodes[i * 3 + j]);
    }
  }

  removed = NULL;
  for(i = 0; i <= 1100; i += 100) {
    printf("Removing nodes not larger than %d\n", i);
    tv_now.tv_usec = i;
    root = Curl_splaygetbest(tv_now, root, &removed);
    while(removed) {
      printf("removed payload %zu[%zu]\n",
             (*(size_t *)removed->payload) / 10,
             (*(size_t *)removed->payload) % 10);
      root = Curl_splaygetbest(tv_now, root, &removed);
    }
  }

  fail_unless(root == NULL, "tree not empty when it should be");

UNITTEST_STOP
