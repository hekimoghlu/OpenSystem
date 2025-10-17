/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 24, 2025.
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
#include <sys/cdefs.h>
#include <stdint.h>
#include <stdbool.h>

__BEGIN_DECLS

#pragma GCC visibility push(hidden)

typedef struct dirstat_fileid_set_s dirstat_fileid_set_s;
typedef dirstat_fileid_set_s *dirstat_fileid_set_t;

dirstat_fileid_set_t _dirstat_fileid_set_create(void);
void _dirstat_fileid_set_destroy(dirstat_fileid_set_t set);
bool _dirstat_fileid_set_add(dirstat_fileid_set_t set, uint64_t fileid);

#pragma GCC visibility pop

__END_DECLS
