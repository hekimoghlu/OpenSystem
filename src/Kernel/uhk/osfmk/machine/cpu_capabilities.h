/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 5, 2025.
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
#ifdef  PRIVATE

#ifndef _MACHINE_CPU_CAPABILITIES_H
#define _MACHINE_CPU_CAPABILITIES_H

#ifdef KERNEL_PRIVATE
#if defined (__i386__) || defined (__x86_64__)
#include "i386/cpu_capabilities.h"
#elif defined (__arm__) || defined (__arm64__)
#include "arm/cpu_capabilities.h"
#else
#error architecture not supported
#endif

#else /* !KERNEL_PRIVATE -- System Framework header */
#if defined (__i386__) || defined(__x86_64__)
#include <i386/cpu_capabilities.h>
#elif defined (__arm__) || defined(__arm64__)
#include <arm/cpu_capabilities.h>
#else
#error architecture not supported
#endif
#endif /* KERNEL_PRIVATE */

#endif /* _MACHINE_CPU_CAPABILITIES_H */
#endif /* PRIVATE */
