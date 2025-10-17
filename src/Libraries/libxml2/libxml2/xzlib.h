/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 24, 2021.
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
#ifndef LIBXML2_XZLIB_H
#define LIBXML2_XZLIB_H
typedef void *xzFile;           /* opaque lzma file descriptor */

xzFile __libxml2_xzopen(const char *path, const char *mode);
xzFile __libxml2_xzdopen(int fd, const char *mode);
int __libxml2_xzread(xzFile file, void *buf, unsigned len);
int __libxml2_xzclose(xzFile file);
int __libxml2_xzcompressed(xzFile f);
#endif /* LIBXML2_XZLIB_H */
