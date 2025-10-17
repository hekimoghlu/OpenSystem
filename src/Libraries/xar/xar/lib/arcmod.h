/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 21, 2025.
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
 * 03-Apr-2005
 * DRI: Rob Braun <bbraun@synack.net>
 */
/*
 * Portions Copyright 2006, Apple Computer, Inc.
 * Christopher Ryan <ryanc@apple.com>
 */


#ifndef _XAR_ARCMOD_H_
#define _XAR_ARCMOD_H_
#include "xar.h"
#include "filetree.h"


typedef int32_t (*arcmod_archive)(xar_t x, xar_file_t f, const char* file, const char *buffer, size_t len);
typedef int32_t (*arcmod_extract)(xar_t x, xar_file_t f, const char* file, char *buffer, size_t len);

struct arcmod {
	arcmod_archive archive;
	arcmod_extract extract;
};

int32_t xar_arcmod_archive(xar_t x, xar_file_t f, const char *file, const char *buffer, size_t len);
int32_t xar_arcmod_extract(xar_t x, xar_file_t f, const char *file, char *buffer, size_t len);

int32_t xar_arcmod_verify(xar_t x, xar_file_t f, xar_progress_callback p);
int32_t xar_check_prop(xar_t x, const char *name);

#endif /* _XAR_ARCMOD_H_ */
