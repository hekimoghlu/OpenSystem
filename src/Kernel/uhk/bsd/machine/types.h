/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 31, 2024.
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
/*
 * Copyright 1995 NeXT Computer, Inc. All rights reserved.
 */
#ifndef _BSD_MACHINE_TYPES_H_
#define _BSD_MACHINE_TYPES_H_

#if defined (__i386__) || defined(__x86_64__)
#include "i386/types.h"
#elif defined (__arm__) || defined (__arm64__)
#include "arm/types.h"
#else
#error architecture not supported
#endif

#endif /* _BSD_MACHINE_TYPES_H_ */
