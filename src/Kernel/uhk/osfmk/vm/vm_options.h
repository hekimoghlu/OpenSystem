/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 19, 2023.
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
#ifndef __VM_VM_OPTIONS_H__
#define __VM_VM_OPTIONS_H__

#define UPL_DEBUG (DEVELOPMENT || DEBUG)
// #define VM_PIP_DEBUG

#define VM_PAGE_BUCKETS_CHECK DEBUG
#if VM_PAGE_BUCKETS_CHECK
#define VM_PAGE_FAKE_BUCKETS 1
#endif /* VM_PAGE_BUCKETS_CHECK */

#define VM_OBJECT_TRACKING 0
#define VM_SCAN_FOR_SHADOW_CHAIN (DEVELOPMENT || DEBUG)

#define VM_OBJECT_ACCESS_TRACKING (DEVELOPMENT || DEBUG)

#define VM_NAMED_ENTRY_DEBUG (DEVELOPMENT || DEBUG)

#define FBDP_DEBUG_OBJECT_NO_PAGER (DEVELOPMENT || DEBUG)

#define PAGE_SLEEP_WITH_INHERITOR (1)

#endif /* __VM_VM_OPTIONS_H__ */
