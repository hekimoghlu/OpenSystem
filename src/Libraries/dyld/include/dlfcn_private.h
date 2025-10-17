/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 21, 2023.
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
#ifndef _DLFCN_PRIVATE_H_
#define _DLFCN_PRIVATE_H_

#include <dlfcn.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * For use in NSCreateObjectFileImageFromMemory()
 */
#define RTLD_UNLOADABLE 0x80000000

/*
 * Internal interface for dlopen; intended to help audit internal use of
 * dlopen.
 */
extern void * dlopen_audited(const char * __path, int __mode) __DYLDDL_UNAVAILABLE;


/*
 * Sometimes dlopen() looks at who called it (such as for @rpath and @loader_path).
 * This SPI allows you to simulate dlopen() being called by other code.
 * Available in macOS 11.0 and iOS 14.0 and later.
 */
extern void* dlopen_from(const char* __path, int __mode, void* __addressInCaller) __DYLDDL_UNAVAILABLE;


#ifdef __cplusplus
}
#endif

#endif /* _DLFCN_PRIVATE_H_ */
