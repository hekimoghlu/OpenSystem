/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 13, 2024.
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
#ifndef __IMG4_H
#define __IMG4_H

#include <os/base.h>
#include <stdint.h>
#include <stdbool.h>
#include <sys/cdefs.h>
#include <sys/kernel_types.h>
#include <sys/types.h>

#define __IMG4_INDIRECT 1
#include <img4/api.h>
#include <img4/firmware.h>

#if !_DARWIN_BUILDING_PROJECT_APPLEIMAGE4
#if IMG4_TARGET_EFI || IMG4_TARGET_SEP
#error "please #include <img4/firmware.h> instead"
#else
#warning "please #include <img4/firmware.h> instead"
#endif
#endif

IMG4_API_DEPRECATED_FALL_2018
typedef uint32_t img4_tag_t;

#endif // __IMG4_H
