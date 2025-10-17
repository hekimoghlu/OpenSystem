/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 28, 2025.
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
#ifndef	_lock_ext_h_
#define	_lock_ext_h_

#if defined(__cplusplus)
extern "C" {
#endif

int __lock_vec_pp __P((DB_ENV *, u_int32_t, u_int32_t, DB_LOCKREQ *, int, DB_LOCKREQ **));
int __lock_vec __P((ENV *, DB_LOCKER *, u_int32_t, DB_LOCKREQ *, int, DB_LOCKREQ **));
int __lock_get_pp __P((DB_ENV *, u_int32_t, u_int32_t, const DBT *, db_lockmode_t, DB_LOCK *));
int __lock_get __P((ENV *, DB_LOCKER *, u_int32_t, const DBT *, db_lockmode_t, DB_LOCK *));
int  __lock_get_internal __P((DB_LOCKTAB *, DB_LOCKER *, u_int32_t, const DBT *, db_lockmode_t, db_timeout_t, DB_LOCK *));
int  __lock_put_pp __P((DB_ENV *, DB_LOCK *));
int  __lock_put __P((ENV *, DB_LOCK *));
int __lock_downgrade __P((ENV *, DB_LOCK *, db_lockmode_t, u_int32_t));
int __lock_locker_is_parent __P((ENV *, DB_LOCKER *, DB_LOCKER *, int *));
int __lock_promote __P((DB_LOCKTAB *, DB_LOCKOBJ *, int *, u_int32_t));
int __lock_detect_pp __P((DB_ENV *, u_int32_t, u_int32_t, int *));
int __lock_detect __P((ENV *, u_int32_t, int *));
int __lock_failchk __P((ENV *));
int __lock_id_pp __P((DB_ENV *, u_int32_t *));
int  __lock_id __P((ENV *, u_int32_t *, DB_LOCKER **));
void __lock_set_thread_id __P((void *, pid_t, db_threadid_t));
int __lock_id_free_pp __P((DB_ENV *, u_int32_t));
int  __lock_id_free __P((ENV *, DB_LOCKER *));
int __lock_id_set __P((ENV *, u_int32_t, u_int32_t));
int __lock_getlocker __P((DB_LOCKTAB *, u_int32_t, int, DB_LOCKER **));
int __lock_getlocker_int __P((DB_LOCKTAB *, u_int32_t, int, DB_LOCKER **));
int __lock_addfamilylocker __P((ENV *, u_int32_t, u_int32_t));
int __lock_freefamilylocker  __P((DB_LOCKTAB *, DB_LOCKER *));
int __lock_freelocker __P((DB_LOCKTAB *, DB_LOCKREGION *, DB_LOCKER *));
int __lock_fix_list __P((ENV *, DBT *, u_int32_t));
int __lock_get_list __P((ENV *, DB_LOCKER *, u_int32_t, db_lockmode_t, DBT *));
void __lock_list_print __P((ENV *, DBT *));
int __lock_env_create __P((DB_ENV *));
void __lock_env_destroy __P((DB_ENV *));
int __lock_get_lk_conflicts __P((DB_ENV *, const u_int8_t **, int *));
int __lock_set_lk_conflicts __P((DB_ENV *, u_int8_t *, int));
int __lock_get_lk_detect __P((DB_ENV *, u_int32_t *));
int __lock_set_lk_detect __P((DB_ENV *, u_int32_t));
int __lock_get_lk_max_locks __P((DB_ENV *, u_int32_t *));
int __lock_set_lk_max_locks __P((DB_ENV *, u_int32_t));
int __lock_get_lk_max_lockers __P((DB_ENV *, u_int32_t *));
int __lock_set_lk_max_lockers __P((DB_ENV *, u_int32_t));
int __lock_get_lk_max_objects __P((DB_ENV *, u_int32_t *));
int __lock_set_lk_max_objects __P((DB_ENV *, u_int32_t));
int __lock_get_lk_partitions __P((DB_ENV *, u_int32_t *));
int __lock_set_lk_partitions __P((DB_ENV *, u_int32_t));
int __lock_get_env_timeout __P((DB_ENV *, db_timeout_t *, u_int32_t));
int __lock_set_env_timeout __P((DB_ENV *, db_timeout_t, u_int32_t));
int __lock_open __P((ENV *, int));
int __lock_env_refresh __P((ENV *));
u_int32_t __lock_region_mutex_count __P((ENV *));
int __lock_stat_pp __P((DB_ENV *, DB_LOCK_STAT **, u_int32_t));
int __lock_stat_print_pp __P((DB_ENV *, u_int32_t));
int  __lock_stat_print __P((ENV *, u_int32_t));
void __lock_printlock __P((DB_LOCKTAB *, DB_MSGBUF *mbp, struct __db_lock *, int));
int __lock_set_timeout __P((ENV *, DB_LOCKER *, db_timeout_t, u_int32_t));
int __lock_set_timeout_internal __P((ENV *, DB_LOCKER *, db_timeout_t, u_int32_t));
int __lock_inherit_timeout __P((ENV *, DB_LOCKER *, DB_LOCKER *));
void __lock_expires __P((ENV *, db_timespec *, db_timeout_t));
int __lock_expired __P((ENV *, db_timespec *, db_timespec *));
u_int32_t __lock_ohash __P((const DBT *));
u_int32_t __lock_lhash __P((DB_LOCKOBJ *));
int __lock_nomem __P((ENV *, const char *));

#if defined(__cplusplus)
}
#endif
#endif /* !_lock_ext_h_ */
