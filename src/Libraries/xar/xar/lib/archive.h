/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 3, 2023.
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


#ifndef _XAR_ARCHIVE_H_
#define _XAR_ARCHIVE_H_
#include <zlib.h>
#include <libxml/hash.h>
#ifdef __APPLE__
#include <CommonCrypto/CommonDigest.h>
#include <CommonCrypto/CommonDigestSPI.h>
#else
#include <openssl/evp.h>
#endif
#include <sys/types.h>
#include <sys/stat.h>
#include "xar.h"
#include "filetree.h"
#include "hash.h"


typedef int (*read_callback)(xar_t, xar_file_t, void *, size_t, void *context);
typedef int (*write_callback)(xar_t, xar_file_t, void *, size_t, void *context);

struct errctx {
	const char *str;
	int         saved_errno;
	xar_file_t  file;
	void       *usrctx;
	xar_t       x;
};

struct __xar_t {
	xar_prop_t props;
	xar_attr_t attrs;      /* archive options, such as rsize */
	const char *prefix;
	const char *ns;
	const char *filler1;
	const char *filler2;
	xar_file_t files;       /* file forest */
	const char *filename;   /* name of the archive we are operating on */
	char *dirname;          /* directory of the archive, used in creation */
	int fd;                 /* open file descriptor for the archive */
	int heap_fd;            /* fd for tmp heap archive, used in creation */
	off_t heap_offset;      /* current offset within the heap */
	off_t heap_len;         /* current length of the heap */
	xar_header_t header;    /* header of the xar archive */
	void *readbuf;          /* buffer for reading/writing compressed toc */
	size_t readbuf_len;     /* length of readbuf */
	size_t offset;          /* offset into readbuf for keeping track
	                         * between callbacks. */
	size_t toc_count;       /* current bytes read of the toc */
	z_stream zs;            /* gz state for compressing/decompressing toc */
	char *path_prefix;      /* used for distinguishing absolute paths */
	err_handler ercallback; /* callback for errors/warnings */
	struct errctx errctx;   /* error callback context */
	xar_subdoc_t subdocs;   /* linked list of subdocs */
	xar_signature_t signatures; /* linked list of signatures */
	int32_t (*attrcopy_to_heap)(xar_t, xar_file_t, xar_prop_t, read_callback, void *);
	int32_t (*attrcopy_from_heap)(xar_t, xar_file_t, xar_prop_t, write_callback, void *);
	int32_t (*heap_to_archive)(xar_t);
	uint64_t last_fileid;       /* unique fileid's in the archive */
	xmlHashTablePtr ino_hash;   /* Hash for looking up hardlinked files (add)*/
	xmlHashTablePtr link_hash;  /* Hash for looking up hardlinked files (extract)*/
	xmlHashTablePtr csum_hash;  /* Hash for looking up checksums of files */
	xar_hash_t toc_hash_ctx;
	int toc_hash_size;			/* size of toc hash that was copied during archive open */
	void *toc_hash;				/* copy of the toc hash copied during archive open */
	int skipwarn;
	struct stat sbcache;
};

#define XAR(x) ((struct __xar_t *)(x))

#endif /* _XAR_ARCHIVE_H_ */
