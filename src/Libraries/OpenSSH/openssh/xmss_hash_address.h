/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 19, 2022.
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

#ifdef WITH_XMSS
/* $OpenBSD: xmss_hash_address.h,v 1.2 2018/02/26 03:56:44 dtucker Exp $ */
/*
hash_address.h version 20160722
Andreas HÃ¼lsing
Joost Rijneveld
Public domain.
*/

#ifdef HAVE_STDINT_H
#include <stdint.h>
#endif

void setLayerADRS(uint32_t adrs[8], uint32_t layer);

void setTreeADRS(uint32_t adrs[8], uint64_t tree);

void setType(uint32_t adrs[8], uint32_t type);

void setKeyAndMask(uint32_t adrs[8], uint32_t keyAndMask);

// OTS

void setOTSADRS(uint32_t adrs[8], uint32_t ots);

void setChainADRS(uint32_t adrs[8], uint32_t chain);

void setHashADRS(uint32_t adrs[8], uint32_t hash);

// L-tree

void setLtreeADRS(uint32_t adrs[8], uint32_t ltree);

// Hash Tree & L-tree

void setTreeHeight(uint32_t adrs[8], uint32_t treeHeight);

void setTreeIndex(uint32_t adrs[8], uint32_t treeIndex);

#endif /* WITH_XMSS */
