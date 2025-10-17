/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 3, 2025.
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
#ifndef DB4_UTILS_H
#define DB4_UTILS_H

#include "db_cxx.h"
#include "mod_db4_export.h"

/* locks */
int env_locks_init();
void env_global_rw_lock();
void env_global_rd_lock();
void env_global_unlock();
void env_wait_for_child_crash();
void env_child_crash();
void env_ok_to_proceed();

void env_rsrc_list_init();

int global_ref_count_increase(char *path);
int global_ref_count_decrease(char *path);
int global_ref_count_get(const char *path);
void global_ref_count_clean();

#endif
/* vim: set ts=4 sts=4 expandtab bs=2 ai fdm=marker: */
