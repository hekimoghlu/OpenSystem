/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 1, 2024.
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
 * Copyright (c) 2010 Miodrag Vallat.
 *
 * Permission to use, copy, modify, and distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#include <lib/libsa/stand.h>

/* where the initrd is loaded */
#define	INITRD_BASE	PHYS_TO_CKSEG0(0x04000000)

/*
 * MD interfaces for MI boot(9)
 */
void	devboot(dev_t, char *);
void	machdep(void);
void	run_loadfile(uint64_t *, int);

/*
 * PMON console
 */
void	pmon_cnprobe(struct consdev *);
void	pmon_cninit(struct consdev *);
int	pmon_cngetc(dev_t);
void	pmon_cnputc(dev_t, int);

/*
 * PMON I/O
 */
int	pmon_iostrategy(void *, int, daddr_t, size_t, void *, size_t *);
int	pmon_ioopen(struct open_file *, ...);
int	pmon_ioclose(struct open_file *);

/*
 * INITRD I/O
 */
int	rd_iostrategy(void *, int, daddr_t, size_t, void *, size_t *);
int	rd_ioopen(struct open_file *, ...);
int	rd_ioclose(struct open_file *);
int	rd_isvalid(void);
void	rd_invalidate(void);

/*
 * INITRD ``filesystem''
 */
int	rdfs_open(char *path, struct open_file *f);
int	rdfs_close(struct open_file *f);
int	rdfs_read(struct open_file *f, void *buf, size_t size, size_t *resid);
int	rdfs_write(struct open_file *f, void *buf, size_t size, size_t *resid);
off_t	rdfs_seek(struct open_file *f, off_t offset, int where);
int	rdfs_stat(struct open_file *f, struct stat *sb);
int	rdfs_readdir(struct open_file *f, char *name);

extern int pmon_argc;
extern int32_t *pmon_argv;
extern int32_t *pmon_envp;
extern int32_t pmon_callvec;

extern char pmon_bootdev[];

extern char *kernelfile;
