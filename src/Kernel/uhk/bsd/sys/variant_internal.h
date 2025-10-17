/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 8, 2022.
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
#ifndef _SYS_VARIANT_INTERNAL_H_
#define _SYS_VARIANT_INTERNAL_H_

__BEGIN_DECLS

enum os_variant_check_status {
	OS_VARIANT_S_UNKNOWN = 0,
	OS_VARIANT_S_NO = 2,
	OS_VARIANT_S_YES = 3
};

/*
 * Bit allocation in kern.osvariant_status (all ranges inclusive):
 * - [0-27] are 2-bit check_status values
 * - [28-31] are 0xF
 * - [32-32+VP_MAX-1] encode variant_property booleans
 * - [48-51] encode the boot mode, if known
 * - [60-62] are 0x7
 */
#define OS_VARIANT_STATUS_INITIAL_BITS 0x70000000F0000000ULL
#define OS_VARIANT_STATUS_BIT_WIDTH 2
#define OS_VARIANT_STATUS_SET 0x2
#define OS_VARIANT_STATUS_MASK 0x3

enum os_variant_status_flags_positions {
	OS_VARIANT_SFP_INTERNAL_CONTENT = 0,
	OS_VARIANT_SFP_INTERNAL_RELEASE_TYPE = 2,
	OS_VARIANT_SFP_INTERNAL_DIAGS_PROFILE = 3,
};

enum os_variant_property {
	OS_VARIANT_PROPERTY_CONTENT,
	OS_VARIANT_PROPERTY_DIAGNOSTICS
};

__END_DECLS

#ifdef KERNEL_PRIVATE

__BEGIN_DECLS

bool os_variant_has_internal_diagnostics(const char *subsystem);

__END_DECLS
#endif /* KERNEL_PRIVATE */

#endif /* _SYS_VARIANT_INTERNAL_H_ */
