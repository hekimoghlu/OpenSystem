/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 23, 2022.
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
#ifndef _QOS_LEGACY_H
#define _QOS_LEGACY_H

#include_next <pthread/qos.h>

#if __DARWIN_C_LEVEL >= __DARWIN_C_FULL

#ifdef __has_include
#if __has_include(<pthread/qos_private.h>)
#include <pthread/qos_private.h>
#endif
#endif

#endif // __DARWIN_C_LEVEL >= __DARWIN_C_FULL

#endif //_QOS_LEGACY_H
