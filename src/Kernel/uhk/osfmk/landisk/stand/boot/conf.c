/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 5, 2022.
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
 * Copyright (c) 2006 Michael Shalayeff
 * All rights reserved.
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF MIND, USE, DATA OR PROFITS, WHETHER IN
 * AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT
 * OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#include <sys/param.h>
#include <libsa.h>
#include <lib/libsa/ufs.h>
#include <lib/libsa/ufs2.h>
#include <dev/cons.h>

const char version[] = "1.11";
#ifdef DEBUG
int	debug = 1;
#endif

struct fs_ops file_system[] = {
	{ ufs_open,    ufs_close,    ufs_read,    ufs_write,    ufs_seek,
	  ufs_stat,    ufs_readdir,  ufs_fchmod },
	{ ufs2_open,   ufs2_close,   ufs2_read,   ufs2_write,   ufs2_seek,
	  ufs2_stat,   ufs2_readdir, ufs2_fchmod }
};
int nfsys = nitems(file_system);

struct devsw	devsw[] = {
	{ "dk", blkdevstrategy, blkdevopen, blkdevclose, noioctl }
};
int ndevs = nitems(devsw);

struct consdev constab[] = {
	{ scif_cnprobe, scif_cninit, scif_cngetc, scif_cnputc },
	{ NULL }
};
struct consdev *cn_tab;

