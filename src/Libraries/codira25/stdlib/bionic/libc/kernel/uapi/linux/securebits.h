/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 15, 2021.
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
#ifndef _UAPI_LINUX_SECUREBITS_H
#define _UAPI_LINUX_SECUREBITS_H
#define issecure_mask(X) (1 << (X))
#define SECUREBITS_DEFAULT 0x00000000
#define SECURE_NOROOT 0
#define SECURE_NOROOT_LOCKED 1
#define SECBIT_NOROOT (issecure_mask(SECURE_NOROOT))
#define SECBIT_NOROOT_LOCKED (issecure_mask(SECURE_NOROOT_LOCKED))
#define SECURE_NO_SETUID_FIXUP 2
#define SECURE_NO_SETUID_FIXUP_LOCKED 3
#define SECBIT_NO_SETUID_FIXUP (issecure_mask(SECURE_NO_SETUID_FIXUP))
#define SECBIT_NO_SETUID_FIXUP_LOCKED (issecure_mask(SECURE_NO_SETUID_FIXUP_LOCKED))
#define SECURE_KEEP_CAPS 4
#define SECURE_KEEP_CAPS_LOCKED 5
#define SECBIT_KEEP_CAPS (issecure_mask(SECURE_KEEP_CAPS))
#define SECBIT_KEEP_CAPS_LOCKED (issecure_mask(SECURE_KEEP_CAPS_LOCKED))
#define SECURE_NO_CAP_AMBIENT_RAISE 6
#define SECURE_NO_CAP_AMBIENT_RAISE_LOCKED 7
#define SECBIT_NO_CAP_AMBIENT_RAISE (issecure_mask(SECURE_NO_CAP_AMBIENT_RAISE))
#define SECBIT_NO_CAP_AMBIENT_RAISE_LOCKED (issecure_mask(SECURE_NO_CAP_AMBIENT_RAISE_LOCKED))
#define SECURE_ALL_BITS (issecure_mask(SECURE_NOROOT) | issecure_mask(SECURE_NO_SETUID_FIXUP) | issecure_mask(SECURE_KEEP_CAPS) | issecure_mask(SECURE_NO_CAP_AMBIENT_RAISE))
#define SECURE_ALL_LOCKS (SECURE_ALL_BITS << 1)
#endif
