/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 9, 2022.
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
 * nfsfh.h - NFS file handle definitions (for portable use)
 *
 * Jeffrey C. Mogul
 * Digital Equipment Corporation
 * Western Research Laboratory
 */

/*
 * Internal representation of dev_t, because different NFS servers
 * that we might be spying upon use different external representations.
 */
typedef struct {
	uint32_t Minor;	/* upper case to avoid clashing with macro names */
	uint32_t Major;
} my_devt;

#define	dev_eq(a,b)	((a.Minor == b.Minor) && (a.Major == b.Major))

/*
 * Many file servers now use a large file system ID.  This is
 * our internal representation of that.
 */
typedef	struct {
	my_devt	Fsid_dev;		/* XXX avoid name conflict with AIX */
	char Opaque_Handle[2 * 32 + 1];
	uint32_t fsid_code;
} my_fsid;

#define	fsid_eq(a,b)	((a.fsid_code == b.fsid_code) &&\
			 dev_eq(a.Fsid_dev, b.Fsid_dev))

extern void Parse_fh(netdissect_options *, const unsigned char *, u_int, my_fsid *, uint32_t *, const char **, const char **, int);
