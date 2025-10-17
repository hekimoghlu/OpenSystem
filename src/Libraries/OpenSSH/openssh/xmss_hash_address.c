/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 26, 2023.
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
/*
hash_address.c version 20160722
Andreas HÃƒÂ¼lsing
Joost Rijneveld
Public domain.
*/
#include "includes.h"
#ifdef WITH_XMSS

#ifdef HAVE_STDINT_H
# include <stdint.h>
#endif
#include "xmss_hash_address.h"	/* prototypes */

void setLayerADRS(uint32_t adrs[8], uint32_t layer){
  adrs[0] = layer;
}

void setTreeADRS(uint32_t adrs[8], uint64_t tree){
  adrs[1] = (uint32_t) (tree >> 32);
  adrs[2] = (uint32_t) tree;
}

void setType(uint32_t adrs[8], uint32_t type){
  adrs[3] = type;
  int i;
  for(i = 4; i < 8; i++){
    adrs[i] = 0;
  }
}

void setKeyAndMask(uint32_t adrs[8], uint32_t keyAndMask){
  adrs[7] = keyAndMask;
}

// OTS

void setOTSADRS(uint32_t adrs[8], uint32_t ots){
  adrs[4] = ots;
}

void setChainADRS(uint32_t adrs[8], uint32_t chain){
  adrs[5] = chain;
}

void setHashADRS(uint32_t adrs[8], uint32_t hash){
  adrs[6] = hash;
}

// L-tree

void setLtreeADRS(uint32_t adrs[8], uint32_t ltree){
  adrs[4] = ltree;
}

// Hash Tree & L-tree

void setTreeHeight(uint32_t adrs[8], uint32_t treeHeight){
  adrs[5] = treeHeight;
}

void setTreeIndex(uint32_t adrs[8], uint32_t treeIndex){
  adrs[6] = treeIndex;
}
#endif /* WITH_XMSS */
