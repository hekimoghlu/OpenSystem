/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 3, 2024.
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
#ifndef PREFERENCES_H
#define PREFERENCES_H

#include <stdbool.h>
#include <unistd.h>

enum { STATMODE_ACCUM = 1, STATMODE_DELTA, STATMODE_EVENT, STATMODE_NON_EVENT };

void top_prefs_init(void);

/*
 * One of:
 * a accumulative mode
 * d delta mode
 * e event mode
 * n non-event mode
 */
bool top_prefs_set_mode(const char *mode);
int top_prefs_get_mode(void);
const char *top_prefs_get_mode_string(void);

void top_prefs_set_sleep(int seconds);
int top_prefs_get_sleep(void);

/* Take a symbolic string such as "cpu" */
bool top_prefs_set_sort(const char *sort);
/* Return one of the TOP_SORT enum values from above. */
int top_prefs_get_sort(void);

bool top_prefs_set_secondary_sort(const char *sort);
int top_prefs_get_secondary_sort(void);

const char *top_prefs_get_sort_string(void);
const char *top_prefs_get_secondary_sort_string(void);

/* This is used to sorting in ascending order (if flag is true). */
void top_prefs_set_ascending(bool flag);

bool top_prefs_get_ascending(void);

void top_prefs_set_frameworks(bool flag);
bool top_prefs_get_frameworks(void);

void top_prefs_set_frameworks_interval(int interval);
int top_prefs_get_frameworks_interval(void);

void top_prefs_set_user(const char *user);

char *top_prefs_get_user(void);

void top_prefs_set_user_uid(uid_t uid);
uid_t top_prefs_get_user_uid(void);

/* Returns true if the comma separated names list is invalid. */
bool top_prefs_set_stats(const char *names);
bool top_prefs_get_stats(int *total, int **array);

int top_prefs_get_samples(void);
void top_prefs_set_samples(int s);

int top_prefs_get_nprocs(void);
void top_prefs_set_nprocs(int n);

void top_prefs_add_pid(pid_t pid);
bool top_prefs_want_pid(pid_t pid);

/* Returns true if the signal string is invalid. */
bool top_prefs_set_signal_string(char *s);
int top_prefs_get_signal(const char **sptr);

void top_prefs_set_logging_mode(bool mode);
bool top_prefs_get_logging_mode(void);

void top_prefs_set_ncols(int limit);
/* Returns true if the ncols has been set. */
bool top_prefs_get_ncols(int *limit);

void top_prefs_set_swap(bool show);
bool top_prefs_get_swap(void);

void top_prefs_set_secondary_ascending(bool flag);
bool top_prefs_get_secondary_ascending(void);

/* memory map reporting */
void top_prefs_set_mmr(bool mmr);
bool top_prefs_get_mmr(void);

#endif /*PREFERENCES_H*/
