/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 22, 2025.
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
 * Structures and definitions which are specific to the PPL incarnation of the
 * kernel's code signing monitor.
 */
#ifndef __IMAGE4_CS_PMAP_H
#define __IMAGE4_CS_PMAP_H

#include <os/base.h>
#include <stdint.h>
#include <sys/types.h>
#include <image4/image4.h>

__BEGIN_DECLS
OS_ASSUME_NONNULL_BEGIN
OS_ASSUME_PTR_ABI_SINGLE_BEGIN

/*!
 * @const IMAGE4_CS_PMAP_DATA_SIZE
 * The size of the memory region which the PPL should reserve on behalf of the
 * GL2 expert.
 */
#define IMAGE4_CS_PMAP_DATA_SIZE (5120u)

OS_ASSUME_PTR_ABI_SINGLE_END
OS_ASSUME_NONNULL_END
__END_DECLS

#endif // __IMAGE4_CS_PMAP_H
