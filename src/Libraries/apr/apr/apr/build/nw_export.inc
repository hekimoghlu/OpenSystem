/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 18, 2024.
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
#include "apr.h"

#undef APR_DECLARE
#undef APR_DECLARE_NONSTD
#undef APR_DECLARE_HOOK
#undef APR_POOL_DECLARE_ACCESSOR
#undef APR_DECLARE_DATA

/* Preprocess all of the standard APR headers. */
#include "apr_allocator.h"
#include "apr_atomic.h"
#include "apr_dso.h"
#include "apr_env.h"
#include "apr_errno.h"
#include "apr_escape.h"
#include "apr_file_info.h"
#include "apr_file_io.h"
#include "apr_fnmatch.h"
#include "apr_general.h"
#include "apr_getopt.h"
#include "apr_global_mutex.h"
#include "apr_hash.h"
#include "apr_inherit.h"
#include "apr_lib.h"
#include "apr_mmap.h"
#include "apr_network_io.h"
#include "apr_poll.h"
#include "apr_pools.h"
#include "apr_portable.h"
#include "apr_proc_mutex.h"
#include "apr_ring.h"
#include "apr_random.h"
#include "apr_shm.h"
#include "apr_signal.h"
#include "apr_skiplist.h"
#include "apr_strings.h"
#include "apr_support.h"
#include "apr_tables.h"
#include "apr_thread_cond.h"
#include "apr_thread_mutex.h"
#include "apr_thread_proc.h"
#include "apr_thread_rwlock.h"
#include "apr_time.h"
#include "apr_user.h"
#include "apr_version.h"
#include "apr_want.h"

#include "nw_apu_export.inc"

