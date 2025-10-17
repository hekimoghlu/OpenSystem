/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 10, 2025.
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
#ifdef __cplusplus
extern "C" {
#endif

/* APPLE: prefix sasl_ to fix conflict with System */ 
extern void * sasl_dlopen(
    const char *path,
    int mode);
extern void * sasl_dlsym(
    void * handle,
    const char *symbol);
extern const char * sasl_dlerror(
    void);
extern int sasl_dlclose(
    void * handle);

#define RTLD_LAZY	0x1
#define RTLD_NOW	0x2
#define RTLD_LOCAL	0x4
#define RTLD_GLOBAL	0x8
#define RTLD_NOLOAD	0x10
#define RTLD_SHARED	0x20	/* not used, the default */
#define RTLD_UNSHARED	0x40
#define RTLD_NODELETE	0x80
#define RTLD_LAZY_UNDEF	0x100

#ifdef __cplusplus
}
#endif
