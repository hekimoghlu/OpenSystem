/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 27, 2024.
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
#ifndef _UAPILINUX_KERNEL_PAGE_FLAGS_H
#define _UAPILINUX_KERNEL_PAGE_FLAGS_H
#define KPF_LOCKED 0
#define KPF_ERROR 1
#define KPF_REFERENCED 2
#define KPF_UPTODATE 3
#define KPF_DIRTY 4
#define KPF_LRU 5
#define KPF_ACTIVE 6
#define KPF_SLAB 7
#define KPF_WRITEBACK 8
#define KPF_RECLAIM 9
#define KPF_BUDDY 10
#define KPF_MMAP 11
#define KPF_ANON 12
#define KPF_SWAPCACHE 13
#define KPF_SWAPBACKED 14
#define KPF_COMPOUND_HEAD 15
#define KPF_COMPOUND_TAIL 16
#define KPF_HUGE 17
#define KPF_UNEVICTABLE 18
#define KPF_HWPOISON 19
#define KPF_NOPAGE 20
#define KPF_KSM 21
#define KPF_THP 22
#define KPF_OFFLINE 23
#define KPF_ZERO_PAGE 24
#define KPF_IDLE 25
#define KPF_PGTABLE 26
#endif
