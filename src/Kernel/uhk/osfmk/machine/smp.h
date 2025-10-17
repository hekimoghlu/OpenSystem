/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 10, 2023.
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
#ifndef _MACHINE_SMP_H
#define _MACHINE_SMP_H

#if defined (__x86_64__)
#include "i386/smp.h"
#elif defined (__arm__) || defined (__arm64__)
#ifdef KERNEL_PRIVATE
/* arm/smp.h isn't installed into the public SDK. */
#include "arm/smp.h"
#endif /* KERNEL_PRIVATE */
#else
#error architecture not supported
#endif

#endif /* _MACHINE_SMP_H */
