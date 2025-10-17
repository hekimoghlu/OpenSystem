/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 19, 2025.
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
#ifndef	_mutex_ext_h_
#define	_mutex_ext_h_

#if defined(__cplusplus)
extern "C" {
#endif

int __mutex_alloc __P((ENV *, int, u_int32_t, db_mutex_t *));
int __mutex_alloc_int __P((ENV *, int, int, u_int32_t, db_mutex_t *));
int __mutex_free __P((ENV *, db_mutex_t *));
int __mutex_free_int __P((ENV *, int, db_mutex_t *));
int __mut_failchk __P((ENV *));
int __db_fcntl_mutex_init __P((ENV *, db_mutex_t, u_int32_t));
int __db_fcntl_mutex_lock __P((ENV *, db_mutex_t));
int __db_fcntl_mutex_unlock __P((ENV *, db_mutex_t));
int __db_fcntl_mutex_destroy __P((ENV *, db_mutex_t));
int __mutex_alloc_pp __P((DB_ENV *, u_int32_t, db_mutex_t *));
int __mutex_free_pp __P((DB_ENV *, db_mutex_t));
int __mutex_lock_pp __P((DB_ENV *, db_mutex_t));
int __mutex_unlock_pp __P((DB_ENV *, db_mutex_t));
int __mutex_get_align __P((DB_ENV *, u_int32_t *));
int __mutex_set_align __P((DB_ENV *, u_int32_t));
int __mutex_get_increment __P((DB_ENV *, u_int32_t *));
int __mutex_set_increment __P((DB_ENV *, u_int32_t));
int __mutex_get_max __P((DB_ENV *, u_int32_t *));
int __mutex_set_max __P((DB_ENV *, u_int32_t));
int __mutex_get_tas_spins __P((DB_ENV *, u_int32_t *));
int __mutex_set_tas_spins __P((DB_ENV *, u_int32_t));
int __db_pthread_mutex_init __P((ENV *, db_mutex_t, u_int32_t));
int __db_pthread_mutex_lock __P((ENV *, db_mutex_t));
int __db_pthread_mutex_unlock __P((ENV *, db_mutex_t));
int __db_pthread_mutex_destroy __P((ENV *, db_mutex_t));
int __mutex_open __P((ENV *, int));
int __mutex_env_refresh __P((ENV *));
void __mutex_resource_return __P((ENV *, REGINFO *));
int __mutex_stat_pp __P((DB_ENV *, DB_MUTEX_STAT **, u_int32_t));
int __mutex_stat_print_pp __P((DB_ENV *, u_int32_t));
int __mutex_stat_print __P((ENV *, u_int32_t));
void __mutex_print_debug_single __P((ENV *, const char *, db_mutex_t, u_int32_t));
void __mutex_print_debug_stats __P((ENV *, DB_MSGBUF *, db_mutex_t, u_int32_t));
void __mutex_set_wait_info __P((ENV *, db_mutex_t, u_int32_t *, u_int32_t *));
void __mutex_clear __P((ENV *, db_mutex_t));
int __db_tas_mutex_init __P((ENV *, db_mutex_t, u_int32_t));
int __db_tas_mutex_lock __P((ENV *, db_mutex_t));
int __db_tas_mutex_unlock __P((ENV *, db_mutex_t));
int __db_tas_mutex_destroy __P((ENV *, db_mutex_t));
int __db_win32_mutex_init __P((ENV *, db_mutex_t, u_int32_t));
int __db_win32_mutex_lock __P((ENV *, db_mutex_t));
int __db_win32_mutex_unlock __P((ENV *, db_mutex_t));
int __db_win32_mutex_destroy __P((ENV *, db_mutex_t));

#if defined(__cplusplus)
}
#endif
#endif /* !_mutex_ext_h_ */
