/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 13, 2021.
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
/*!
 * @header
 * Kernel-private interfaces to link the upward-exported AppleImage4 API. This
 * header serves as an umbrella header to enforce inclusion ordering among its
 * associated headers.
 *
 * This file is intended for use in the xnu project.
 */
#ifndef __IMAGE4_DLXK_H
#define __IMAGE4_DLXK_H

#define __IMAGE4_XNU_INDIRECT 1
#include <image4/image4.h>
#include <image4/types.h>
#include <image4/coprocessor.h>
#include <image4/environment.h>
#include <image4/trust.h>
#include <image4/trust_evaluation.h>
#include <image4/cs/traps.h>

#if XNU_KERNEL_PRIVATE
#include <libkern/image4/interface.h>
#include <libkern/image4/api.h>
#else
#include <image4/dlxk/interface.h>
#endif

__BEGIN_DECLS
OS_ASSUME_NONNULL_BEGIN
OS_ASSUME_PTR_ABI_SINGLE_BEGIN

#pragma mark Definitions
/*!
 * @const IMAGE4_DLXK_VERSION
 * The API version of the upward-ish-linked kernel interface structure. The
 * kernel cannot directly call into kext exports, so the kext instead provides a
 * structure of function pointers and registers that structure with the kernel
 * at boot.
 */
#define IMAGE4_DLXK_VERSION (2u)

#pragma mark KPI
/*!
 * @function image4_dlxk_link
 * Links the interface exported by the AppleImage4 kext via the given structure
 * so that the kernel-proper can use it via the trampolines provided in
 * image4/dlxk/api.h.
 *
 * @param dlxk
 * The interface to link.
 *
 * @discussion
 * This routine may only be called once and must be called prior to machine
 * lockdown.
 */
OS_EXPORT OS_NONNULL1
void
image4_dlxk_link(const image4_dlxk_interface_t *dlxk);

/*!
 * @function image4_dlxk_get
 * Returns the interface structure which was linked at boot.
 *
 * @param v
 * The minimum required version. If the structure's version does not satisfy
 * this constraint, NULL is returned.
 *
 * @result
 * The interface structure which was linked at boot. If no structure was
 * registered at boot, or if the registered structure's version is less than
 * the version specified, NULL is returned.
 */
OS_EXPORT OS_WARN_RESULT
const image4_dlxk_interface_t *_Nullable
image4_dlxk_get(image4_struct_version_t v);

OS_ASSUME_PTR_ABI_SINGLE_END
OS_ASSUME_NONNULL_END
__END_DECLS

#endif // __IMAGE4_DLXK_H
