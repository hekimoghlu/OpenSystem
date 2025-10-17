/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 16, 2023.
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
#include <stdio.h>
#include <stdlib.h>
#include "memcheck.h"

/* Structures to keep track of what has been allocated.  Speed isn't a
 * big issue for the tests this is required for, so we will use a
 * doubly-linked list to make deletion easier.
 */

typedef struct allocation_entry {
  struct allocation_entry *next;
  struct allocation_entry *prev;
  void *allocation;
  size_t num_bytes;
} AllocationEntry;

static AllocationEntry *alloc_head = NULL;
static AllocationEntry *alloc_tail = NULL;

static AllocationEntry *find_allocation(const void *ptr);

/* Allocate some memory and keep track of it. */
void *
tracking_malloc(size_t size) {
  AllocationEntry *const entry
      = (AllocationEntry *)malloc(sizeof(AllocationEntry));

  if (entry == NULL) {
    printf("Allocator failure\n");
    return NULL;
  }
  entry->num_bytes = size;
  entry->allocation = malloc(size);
  if (entry->allocation == NULL) {
    free(entry);
    return NULL;
  }
  entry->next = NULL;

  /* Add to the list of allocations */
  if (alloc_head == NULL) {
    entry->prev = NULL;
    alloc_head = alloc_tail = entry;
  } else {
    entry->prev = alloc_tail;
    alloc_tail->next = entry;
    alloc_tail = entry;
  }

  return entry->allocation;
}

static AllocationEntry *
find_allocation(const void *ptr) {
  AllocationEntry *entry;

  for (entry = alloc_head; entry != NULL; entry = entry->next) {
    if (entry->allocation == ptr) {
      return entry;
    }
  }
  return NULL;
}

/* Free some memory and remove the tracking for it */
void
tracking_free(void *ptr) {
  AllocationEntry *entry;

  if (ptr == NULL) {
    /* There won't be an entry for this */
    return;
  }

  entry = find_allocation(ptr);
  if (entry != NULL) {
    /* This is the relevant allocation.  Unlink it */
    if (entry->prev != NULL)
      entry->prev->next = entry->next;
    else
      alloc_head = entry->next;
    if (entry->next != NULL)
      entry->next->prev = entry->prev;
    else
      alloc_tail = entry->next;
    free(entry);
  } else {
    printf("Attempting to free unallocated memory at %p\n", ptr);
  }
  free(ptr);
}

/* Reallocate some memory and keep track of it */
void *
tracking_realloc(void *ptr, size_t size) {
  AllocationEntry *entry;

  if (ptr == NULL) {
    /* By definition, this is equivalent to malloc(size) */
    return tracking_malloc(size);
  }
  if (size == 0) {
    /* By definition, this is equivalent to free(ptr) */
    tracking_free(ptr);
    return NULL;
  }

  /* Find the allocation entry for this memory */
  entry = find_allocation(ptr);
  if (entry == NULL) {
    printf("Attempting to realloc unallocated memory at %p\n", ptr);
    entry = (AllocationEntry *)malloc(sizeof(AllocationEntry));
    if (entry == NULL) {
      printf("Reallocator failure\n");
      return NULL;
    }
    entry->allocation = realloc(ptr, size);
    if (entry->allocation == NULL) {
      free(entry);
      return NULL;
    }

    /* Add to the list of allocations */
    entry->next = NULL;
    if (alloc_head == NULL) {
      entry->prev = NULL;
      alloc_head = alloc_tail = entry;
    } else {
      entry->prev = alloc_tail;
      alloc_tail->next = entry;
      alloc_tail = entry;
    }
  } else {
    void *const reallocated = realloc(ptr, size);
    if (reallocated == NULL) {
      return NULL;
    }
    entry->allocation = reallocated;
  }

  entry->num_bytes = size;
  return entry->allocation;
}

int
tracking_report(void) {
  AllocationEntry *entry;

  if (alloc_head == NULL)
    return 1;

  /* Otherwise we have allocations that haven't been freed */
  for (entry = alloc_head; entry != NULL; entry = entry->next) {
    printf("Allocated %lu bytes at %p\n", (long unsigned)entry->num_bytes,
           entry->allocation);
  }
  return 0;
}

