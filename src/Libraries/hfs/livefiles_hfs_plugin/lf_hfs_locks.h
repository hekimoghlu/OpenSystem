/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 3, 2024.
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
#ifndef lf_hfs_locks_h
#define lf_hfs_locks_h

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>

typedef enum
{
    LCK_RW_TYPE_SHARED,
    LCK_RW_TYPE_EXCLUSIVE
    
} lck_rwlock_type_e;

typedef uint8_t     lck_attr_t;
typedef uint8_t     lck_grp_attr_t;
typedef uint8_t     lck_grp_t;

// Read/Write locks.
void        lf_lck_rw_init                     ( pthread_rwlock_t* lck );
void        lf_lck_rw_destroy                  ( pthread_rwlock_t* lck );
void        lf_lck_rw_unlock_shared            ( pthread_rwlock_t* lck );
void        lf_lck_rw_lock_shared              ( pthread_rwlock_t* lck );
void        lf_lck_rw_lock_exclusive           ( pthread_rwlock_t* lck );
void        lf_lck_rw_unlock_exclusive         ( pthread_rwlock_t* lck );
bool        lf_lck_rw_try_lock         ( pthread_rwlock_t* lck, lck_rwlock_type_e which );
void        lf_lck_rw_lock_exclusive_to_shared ( pthread_rwlock_t* lck);
bool        lf_lck_rw_lock_shared_to_exclusive ( pthread_rwlock_t* lck);

// Mutex locks.
void lf_lck_mtx_init           ( pthread_mutex_t* lck );
void lf_lck_mtx_destroy        ( pthread_mutex_t *lck );
void lf_lck_mtx_lock           ( pthread_mutex_t* lck );
void lf_lck_mtx_unlock         ( pthread_mutex_t* lck );
void lf_lck_mtx_lock_spin      ( pthread_mutex_t *lck );
int lf_lck_mtx_try_lock        ( pthread_mutex_t *lck );
void lf_lck_mtx_convert_spin   ( pthread_mutex_t *lck );

//Cond
void lf_cond_destroy( pthread_cond_t* cond );
void lf_cond_init( pthread_cond_t* cond );
int  lf_cond_wait_relative(pthread_cond_t *pCond, pthread_mutex_t *pMutex, struct timespec *pTime);
void lf_cond_wakeup(pthread_cond_t *pCond);

// Spin locks.
void    lf_lck_spin_init       ( pthread_mutex_t *lck );
void    lf_lck_spin_destroy    ( pthread_mutex_t *lck );
void    lf_lck_spin_lock       ( pthread_mutex_t *lck );
void    lf_lck_spin_unlock     ( pthread_mutex_t *lck );

// Init
lck_attr_t         *lf_lck_attr_alloc_init     ( void );
lck_grp_attr_t     *lf_lck_grp_attr_alloc_init ( void );
lck_grp_t          *lf_lck_grp_alloc_init      ( void );

#endif /* lf_hfs_locks_h */
