/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 16, 2022.
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
 * Rob Braun <bbraun@synack.net>
 * 21-Apr-2004
 * Copyright (c) 2004 Rob Braun.  All rights reserved.
 */
/*
 * Portions Copyright 2006, Apple Computer, Inc.
 * Christopher Ryan <ryanc@apple.com>
*/

#ifndef _XAR_DATA_H_
#define _XAR_DATA_H_

struct _data_context{
    xar_progress_callback progress;
    int fd;
    void *buffer;
    size_t length;
    off_t offset;
    off_t total;
};

#define DATA_CONTEXT(x) ((struct _data_context*)(x))


int32_t xar_data_archive(xar_t x, xar_file_t f, const char* file, const char *buffer, size_t len);
int32_t xar_data_extract(xar_t x, xar_file_t f, const char* file, char *buffer, size_t len);

int32_t xar_data_verify(xar_t x, xar_file_t f, xar_progress_callback p);
#endif /* _XAR_DATA_H_ */
