/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 1, 2025.
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
#include "sjeng.h"
#include "protos.h"
#include "extvars.h"

typedef struct  
{
uint32_t stored_hash;
uint32_t hold_hash;
int32_t score;
} ECacheType;

/*ECacheType ECache[ECACHESIZE];*/
ECacheType *ECache;

uint32_t ECacheProbes;
uint32_t ECacheHits;

void storeECache(int32_t score)
{
  int index;

  index = hash % ECacheSize;

  ECache[index].stored_hash = hash;
  ECache[index].hold_hash = hold_hash;
  ECache[index].score = score;
}

void checkECache(int32_t *score, int *in_cache)
{
  int index;

  ECacheProbes++;

  index = hash % ECacheSize;

  if(ECache[index].stored_hash == hash &&
	  ECache[index].hold_hash == hold_hash)
    
    {
      ECacheHits++;  

      *in_cache = 1;
      *score = ECache[index].score;
    }
}

void reset_ecache(void)
{
  memset(ECache, 0, sizeof(ECacheType)*ECacheSize);
  return;
}

void alloc_ecache(void)
{
  ECache = (ECacheType*)malloc(sizeof(ECacheType)*ECacheSize);

  if (ECache == NULL)
  {
    printf("Out of memory allocating ECache.\n");
    exit(EXIT_FAILURE);
  }
  
  printf("Allocated %u eval cache entries, totalling %u bytes.\n",
		 (uint32_t)ECacheSize, (uint32_t)(sizeof(ECacheType)*ECacheSize));
  return;
}

void free_ecache(void)
{
  free(ECache);
  return;
}
