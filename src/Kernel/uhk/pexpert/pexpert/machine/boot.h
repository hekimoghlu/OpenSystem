/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 23, 2023.
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
#ifndef _PEXPERT_MACHINE_BOOT_H
#define _PEXPERT_MACHINE_BOOT_H

#if defined (__i386__) || defined(__x86_64__)
#include "pexpert/i386/boot.h"
#elif defined (__arm64__)
#ifdef PRIVATE
/* pexpert/arm64/boot.h isn't installed into the public SDK. */
#include "pexpert/arm64/boot.h"
#endif /* PRIVATE */
#elif defined (__arm__)
/*
 * This file is DEPRECATED for arm architectures preceeding version 8.
 */
#ifdef PRIVATE
/* pexpert/arm/boot.h isn't installed into the public SDK. */
#include "pexpert/arm/boot.h"
#endif /* PRIVATE */
#else
#error architecture not supported
#endif

#endif /* _PEXPERT_MACHINE_BOOT_H */
