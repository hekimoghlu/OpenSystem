/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 26, 2022.
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
#define IOPGSHIFT	11
#define	IONBPG		(1 << IOPGSHIFT)
#define IOPGOFSET	(IONBPG - 1)

struct disklabel;
struct hppa_dev {
	dev_t	bootdev;
	struct pz_device *pz_dev;	/* device descriptor */
	daddr32_t fsoff;			/* offset to the file system */
	daddr32_t	last_blk;		/* byte offset for last read blk */
	size_t	last_read;		/* amount read last time */
	struct disklabel *label;
	/* buffer to cache data (aligned properly) */
	char	*buf;
	char	ua_buf[IODC_IOSIZ + IODC_MINIOSIZ];
};

#ifdef PDCDEBUG
#define	DEVPATH_PRINT(dp) \
	printf("%x, %d.%d.%d.%d.%d.%d, 0x%x, %x.%x.%x.%x.%x.%x\n", \
	       (dp)->dp_flags, (dp)->dp_bc[0], (dp)->dp_bc[1], (dp)->dp_bc[2], \
	       (dp)->dp_bc[3], (dp)->dp_bc[4], (dp)->dp_bc[5], (dp)->dp_mod, \
	       (dp)->dp_layers[0], (dp)->dp_layers[1], (dp)->dp_layers[2], \
	       (dp)->dp_layers[3], (dp)->dp_layers[4], (dp)->dp_layers[5]);
#define	PZDEV_PRINT(dp) \
	printf("devpath={%x, %d.%d.%d.%d.%d.%d, 0x%x, %x.%x.%x.%x.%x.%x}," \
	       "\n\thpa=%p, spa=%p, io=%p, class=%u\n", \
	       (dp)->pz_flags, (dp)->pz_bc[0], (dp)->pz_bc[1], (dp)->pz_bc[2], \
	       (dp)->pz_bc[3], (dp)->pz_bc[4], (dp)->pz_bc[5], (dp)->pz_mod, \
	       (dp)->pz_layers[0], (dp)->pz_layers[1], (dp)->pz_layers[2], \
	       (dp)->pz_layers[3], (dp)->pz_layers[4], (dp)->pz_layers[5], \
	       (dp)->pz_hpa, (dp)->pz_spa, (dp)->pz_iodc_io, (dp)->pz_class);
#endif

extern pdcio_t pdc;
extern int pdcbuf[];			/* PDC returns, pdc.c */

int iodc_rw(char *, u_int, u_int, int func, struct pz_device *);
const char *dk_disklabel(struct hppa_dev *dp, struct disklabel *label);

