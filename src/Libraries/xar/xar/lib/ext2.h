/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 6, 2024.
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
 * 26-Oct-2004
 * Copyright (c) 2004 Rob Braun.  All rights reserved.
 */
/*
 * Portions Copyright 2006, Apple Computer, Inc.
 * Christopher Ryan <ryanc@apple.com>
*/

#ifndef _XAR_EXT2_H_
#define _XAR_EXT2_H_
#define XAR_ATTR_FORK "attribute"
int xar_ext2attr_archive(xar_t x, xar_file_t f, const char* file, const char *buffer, size_t len);
int xar_ext2attr_extract(xar_t x, xar_file_t f, const char* file, char *buffer, size_t len);
#endif /* _XAR_EXT2_H_ */
