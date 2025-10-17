/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 22, 2024.
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

// Copyright (c) 2024 Apple Inc.  All rights reserved.

/*
 * Listing of scheduler-related headers that are exported outside of the kernel.
 */

#include <kern/bits.h>
#include <kern/coalition.h>
#include <kern/policy_internal.h>
#include <kern/processor.h>
#include <kern/sched_amp_common.h>
#include <kern/sched_prim.h>
#include <kern/sched_urgency.h>
#include <kern/thread_call.h>
#include <kern/timer_call.h>
#include <kern/waitq.h>
#define CONFIG_THREAD_GROUPS 1
typedef void *cluster_type_t;
#include <kern/thread_group.h>
#include <kern/work_interval.h>
