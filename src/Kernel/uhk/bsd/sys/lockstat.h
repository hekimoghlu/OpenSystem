/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 2, 2025.
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
/*
 * Copyright 1997-2003 Sun Microsystems, Inc.  All rights reserved.
 * Use is subject to license terms.
 */

#ifndef _SYS_LOCKSTAT_H
#define _SYS_LOCKSTAT_H

#ifdef  __cplusplus
extern "C" {
#endif

/*
 * Name the various locking functions...
 */
#define LS_LCK_MTX_LOCK_SPIN            "lck_mtx_lock_spin"
#define LS_LCK_MTX_LOCK                 "lck_mtx_lock"
#define LS_LCK_MTX_TRY_LOCK_SPIN        "lck_mtx_try_lock_spin"
#define LS_LCK_MTX_TRY_LOCK             "lck_mtx_try_lock"
#define LS_LCK_MTX_UNLOCK               "lck_mtx_unlock"

#define LS_LCK_SPIN_LOCK                "lck_spin_lock"
#define LS_LCK_SPIN_TRY_LOCK            "lck_spin_try_lock"
#define LS_LCK_SPIN_UNLOCK              "lck_spin_unlock"
#define LS_LCK_RW_LOCK_SHARED           "lck_rw_lock_shared"
#define LS_LCK_RW_LOCK_EXCL             "lck_rw_lock_exclusive"
#define LS_LCK_RW_DONE                  "lck_rw_done"
#define LS_LCK_RW_TRY_LOCK_EXCL         "lck_rw_try_lock_exclusive"
#define LS_LCK_RW_TRY_LOCK_SHARED       "lck_rw_try_lock_shared"
#define LS_LCK_RW_LOCK_SHARED_TO_EXCL   "lck_rw_lock_shared_to_exclusive"
#define LS_LCK_RW_LOCK_EXCL_TO_SHARED   "lck_rw_lock_exclusive_to_shared"
#define LS_LCK_TICKET_LOCK              "lck_ticket_lock"
#define LS_LCK_TICKET_UNLOCK            "lck_ticket_unlock"


#define LS_ACQUIRE                      "acquire"
#define LS_RELEASE                      "release"
#define LS_SPIN                         "spin"
#define LS_BLOCK                        "block"
#define LS_UPGRADE                      "upgrade"
#define LS_DOWNGRADE                    "downgrade"

#define LS_TYPE_ADAPTIVE                "adaptive" /* this really means "mutex" */
#define LS_TYPE_SPIN                    "spin"
#define LS_TYPE_RW                      "rw"
#define LS_TYPE_TICKET                  "ticket"

#define LSA_ACQUIRE                     (LS_TYPE_ADAPTIVE "-" LS_ACQUIRE)
#define LSA_RELEASE                     (LS_TYPE_ADAPTIVE "-" LS_RELEASE)
#define LSA_SPIN                        (LS_TYPE_ADAPTIVE "-" LS_SPIN)
#define LSA_BLOCK                       (LS_TYPE_ADAPTIVE "-" LS_BLOCK)
#define LSS_ACQUIRE                     (LS_TYPE_SPIN "-" LS_ACQUIRE)
#define LSS_RELEASE                     (LS_TYPE_SPIN "-" LS_RELEASE)
#define LSS_SPIN                        (LS_TYPE_SPIN "-" LS_SPIN)
#define LSR_ACQUIRE                     (LS_TYPE_RW "-" LS_ACQUIRE)
#define LSR_RELEASE                     (LS_TYPE_RW "-" LS_RELEASE)
#define LSR_BLOCK                       (LS_TYPE_RW "-" LS_BLOCK)
#define LSR_SPIN                        (LS_TYPE_RW "-" LS_SPIN)
#define LSR_UPGRADE                     (LS_TYPE_RW "-" LS_UPGRADE)
#define LSR_DOWNGRADE                   (LS_TYPE_RW "-" LS_DOWNGRADE)
#define LST_ACQUIRE                     (LS_TYPE_TICKET "-" LS_ACQUIRE)
#define LST_RELEASE                     (LS_TYPE_TICKET "-" LS_RELEASE)
#define LST_SPIN                        (LS_TYPE_TICKET "-" LS_SPIN)

#ifdef  __cplusplus
}
#endif

#endif  /* _SYS_LOCKSTAT_H */
