/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 1, 2024.
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
#ifndef lf_hfs_rangelist_h
#define lf_hfs_rangelist_h

#include <stdio.h>
#include <sys/queue.h>

TAILQ_HEAD(rl_head, rl_entry);

struct rl_entry {
    TAILQ_ENTRY(rl_entry) rl_link;
    off_t rl_start;
    off_t rl_end;
};

enum rl_overlaptype {
    RL_NOOVERLAP = 0,        /* 0 */
    RL_MATCHINGOVERLAP,      /* 1 */
    RL_OVERLAPCONTAINSRANGE, /* 2 */
    RL_OVERLAPISCONTAINED,   /* 3 */
    RL_OVERLAPSTARTSBEFORE,  /* 4 */
    RL_OVERLAPENDSAFTER      /* 5 */
};

#define RL_INFINITY INT64_MAX

void rl_init(struct rl_head *rangelist);
enum rl_overlaptype rl_overlap(const struct rl_entry *range, off_t start, off_t end);
void rl_remove(off_t start, off_t end, struct rl_head *rangelist);
off_t rl_len(const struct rl_entry *range);
void rl_remove_all(struct rl_head *rangelist);
enum rl_overlaptype rl_scan(struct rl_head *rangelist, off_t start, off_t end, struct rl_entry **overlap);
void rl_add(off_t start, off_t end, struct rl_head *rangelist);
void rl_subtract(struct rl_entry *a, const struct rl_entry *b);
struct rl_entry rl_make(off_t start, off_t end);

#endif /* lf_hfs_rangelist_h */
