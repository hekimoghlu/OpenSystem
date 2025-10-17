/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 26, 2021.
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
/*	$NetBSD: hfs.h,v 1.1 2000/11/14 11:25:35 tsubai Exp $	*/

int hfs_open(char *, struct open_file *);
int hfs_close(struct open_file *);
int hfs_read(struct open_file *, void *, size_t, size_t *);
int hfs_write(struct open_file *, void *, size_t, size_t *);
off_t hfs_seek(struct open_file *, off_t, int);
int hfs_stat(struct open_file *, struct stat *);
int hfs_readdir(struct open_file *, char *);
