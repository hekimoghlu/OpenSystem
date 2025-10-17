/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 5, 2022.
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
#ifndef MOD_DB4_SEM_UTILS_H
#define MOD_DB4_SEM_UTILS_H

extern int md4_sem_create(int semnum, unsigned short *start);
extern void md4_sem_destroy(int semid);
extern void md4_sem_lock(int semid, int semnum);
extern void md4_sem_unlock(int semid, int semnum);
extern void md4_sem_wait_for_zero(int semid, int semnum);
extern void md4_sem_set(int semid, int semnum, int value);
extern int md4_sem_get(int semid, int semnum);

/* vim: set ts=4 sts=4 expandtab bs=2 ai : */
#endif
