/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 25, 2023.
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
#ifndef _XT_SCTP_H_
#define _XT_SCTP_H_
#include <linux/types.h>
#define XT_SCTP_SRC_PORTS 0x01
#define XT_SCTP_DEST_PORTS 0x02
#define XT_SCTP_CHUNK_TYPES 0x04
#define XT_SCTP_VALID_FLAGS 0x07
struct xt_sctp_flag_info {
  __u8 chunktype;
  __u8 flag;
  __u8 flag_mask;
};
#define XT_NUM_SCTP_FLAGS 4
struct xt_sctp_info {
  __u16 dpts[2];
  __u16 spts[2];
  __u32 chunkmap[256 / sizeof(__u32)];
#define SCTP_CHUNK_MATCH_ANY 0x01
#define SCTP_CHUNK_MATCH_ALL 0x02
#define SCTP_CHUNK_MATCH_ONLY 0x04
  __u32 chunk_match_type;
  struct xt_sctp_flag_info flag_info[XT_NUM_SCTP_FLAGS];
  int flag_count;
  __u32 flags;
  __u32 invflags;
};
#define bytes(type) (sizeof(type) * 8)
#define SCTP_CHUNKMAP_SET(chunkmap,type) do { (chunkmap)[type / bytes(__u32)] |= 1u << (type % bytes(__u32)); } while(0)
#define SCTP_CHUNKMAP_CLEAR(chunkmap,type) do { (chunkmap)[type / bytes(__u32)] &= ~(1u << (type % bytes(__u32))); } while(0)
#define SCTP_CHUNKMAP_IS_SET(chunkmap,type) \
({ ((chunkmap)[type / bytes(__u32)] & (1u << (type % bytes(__u32)))) ? 1 : 0; \
})
#define SCTP_CHUNKMAP_RESET(chunkmap) memset((chunkmap), 0, sizeof(chunkmap))
#define SCTP_CHUNKMAP_SET_ALL(chunkmap) memset((chunkmap), ~0U, sizeof(chunkmap))
#define SCTP_CHUNKMAP_COPY(destmap,srcmap) memcpy((destmap), (srcmap), sizeof(srcmap))
#define SCTP_CHUNKMAP_IS_CLEAR(chunkmap) __sctp_chunkmap_is_clear((chunkmap), ARRAY_SIZE(chunkmap))
#define SCTP_CHUNKMAP_IS_ALL_SET(chunkmap) __sctp_chunkmap_is_all_set((chunkmap), ARRAY_SIZE(chunkmap))
#endif
