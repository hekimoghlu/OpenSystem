/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 9, 2024.
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


#include "config.h"
#ifndef HAVE_ASPRINTF
#include "asprintf.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <zlib.h>
#include <errno.h>
#include "xar.h"
#include "filetree.h"
#include "io.h"

struct _gzip_context{
	uint8_t		gzipcompressed;
	uint64_t        count;
	z_stream	z;
};

#define GZIP_CONTEXT(x) ((struct _gzip_context *)(*x))

int xar_gzip_fromheap_done(xar_t x, xar_file_t f, xar_prop_t p, void **context) {

	if( !context || !GZIP_CONTEXT(context) )
		return 0;

	if( GZIP_CONTEXT(context)->gzipcompressed){
		inflateEnd(&GZIP_CONTEXT(context)->z);
	}

	/* free the context */
	free(GZIP_CONTEXT(context));
	*context = NULL;
	
	return 0;
}

int xar_gzip_fromheap_in(xar_t x, xar_file_t f, xar_prop_t p, void **in, size_t *inlen, void **context) {
	const char *opt;
	void *out = NULL;
	size_t outlen, offset = 0;
	int r;
	xar_prop_t tmpp;

	/* on first run, we init the context and check the compression type */
	if( !GZIP_CONTEXT(context) ) {
		*context = calloc(1,sizeof(struct _gzip_context));
		
		opt = NULL;
		tmpp = xar_prop_pget(p, "encoding");
		if( tmpp )
			opt = xar_attr_pget(f, tmpp, "style");
		if( !opt ) return 0;
		if( strcmp(opt, "application/x-gzip") != 0 ) return 0;
		
		inflateInit(&GZIP_CONTEXT(context)->z);
		GZIP_CONTEXT(context)->gzipcompressed = 1;
	}else if( !GZIP_CONTEXT(context)->gzipcompressed ){
		/* once the context has been initialized, then we have already
		   checked the compression type, so we need only check if we
		   actually are compressed */
		return 0;
	}

	outlen = *inlen;

	GZIP_CONTEXT(context)->z.next_in = *in;
	GZIP_CONTEXT(context)->z.avail_in = *inlen;
	GZIP_CONTEXT(context)->z.next_out = out;
	GZIP_CONTEXT(context)->z.avail_out = 0;

	while( GZIP_CONTEXT(context)->z.avail_in != 0 ) {
		size_t newlen = outlen * 2;
		if (newlen > outlen)
		{
			outlen = newlen;
		}
		else
		{
			/* Overflow: Someone has somehow malloced over 2^64 bits of ram. */
			xar_err_new(x);
			xar_err_set_file(x, f);
			xar_err_set_string(x, "Error decompressing file. Decompressor has overflowed.");
			xar_err_callback(x, XAR_SEVERITY_FATAL, XAR_ERR_ARCHIVE_EXTRACTION);
			free(out);
			out = NULL;
			return -1;
		}
		
		void *re_out = realloc(out, outlen);
		if( re_out == NULL )
		{
			xar_err_new(x);
			xar_err_set_file(x, f);
			xar_err_set_string(x, "Error decompressing file. realloc failed");
			xar_err_callback(x, XAR_SEVERITY_FATAL, XAR_ERR_ARCHIVE_EXTRACTION);
			free(out);
			out = NULL;
			return -1;
		}
		
		out = re_out;

		GZIP_CONTEXT(context)->z.next_out = ((unsigned char *)out) + offset;
		GZIP_CONTEXT(context)->z.avail_out = outlen - offset;

		r = inflate(&(GZIP_CONTEXT(context)->z), Z_NO_FLUSH);
		if( (r != Z_OK) && (r != Z_STREAM_END) ) {
			xar_err_new(x);
			xar_err_set_file(x, f);
			xar_err_set_string(x, "Error decompressing file");
			xar_err_callback(x, XAR_SEVERITY_FATAL, XAR_ERR_ARCHIVE_EXTRACTION);
			free(out);
			out = NULL;
			return -1;
		}
		
		offset += outlen - offset - GZIP_CONTEXT(context)->z.avail_out;
		
	}

	free(*in);
	*in = out;
	*inlen = offset;
	return 0;
}

int xar_gzip_toheap_done(xar_t x, xar_file_t f, xar_prop_t p, void **context) {
	xar_prop_t tmpp;
	
	if( GZIP_CONTEXT(context)->gzipcompressed){
		deflateEnd(&GZIP_CONTEXT(context)->z);		
		
		if( GZIP_CONTEXT(context)->count ) {
			tmpp = xar_prop_pset(f, p, "encoding", NULL);
			if( tmpp )
				xar_attr_pset(f, tmpp, "style", "application/x-gzip");
		}
	}

	/* free the context */
	free(GZIP_CONTEXT(context));
	*context = NULL;
	
	return 0;
}

int32_t xar_gzip_toheap_in(xar_t x, xar_file_t f, xar_prop_t p, void **in, size_t *inlen, void **context) {
	void *out = NULL;
	size_t outlen, offset = 0;
	int r;
	const char *opt;

	/* on first run, we init the context and check the compression type */
	if( !GZIP_CONTEXT(context) ) {
		int level = Z_BEST_COMPRESSION;
		*context = calloc(1,sizeof(struct _gzip_context));
		
		opt = xar_opt_get(x, XAR_OPT_COMPRESSION);
		if( !opt )
			return 0;
		
		if( strcmp(opt, XAR_OPT_VAL_GZIP) != 0 )
			return 0;

		opt = xar_opt_get(x, XAR_OPT_COMPRESSIONARG);
		if( opt ) {
			int tmp;
			errno = 0;
			tmp = strtol(opt, NULL, 10);
			if( errno == 0 ) {
				if( (level >= 0) && (level <= 9) )
					level = tmp;
			}
		}
		
		deflateInit(&GZIP_CONTEXT(context)->z, level);
		GZIP_CONTEXT(context)->gzipcompressed = 1;
		if( *inlen == 0 )
			return 0;
	}else if( !GZIP_CONTEXT(context)->gzipcompressed ){
		/* once the context has been initialized, then we have already
		checked the compression type, so we need only check if we
		actually are compressed */
		return 0;
	}
	
	outlen = *inlen/2;
	if(outlen == 0) outlen = 1024;
	GZIP_CONTEXT(context)->z.next_in = *in;
	GZIP_CONTEXT(context)->z.avail_in = *inlen;
	GZIP_CONTEXT(context)->z.next_out = out;
	GZIP_CONTEXT(context)->z.avail_out = 0;

	if( *inlen != 0 ) {
		do {
			size_t newlen = outlen * 2;
			if (newlen > outlen)
				outlen = newlen;
			else
				abort();	/* Someone has somehow malloced over 2^64 bits of ram. */
			
			out = realloc(out, outlen);
			if( out == NULL ) abort();

			GZIP_CONTEXT(context)->z.next_out = ((unsigned char *)out) + offset;
			GZIP_CONTEXT(context)->z.avail_out = outlen - offset;

			r = deflate(&GZIP_CONTEXT(context)->z, Z_NO_FLUSH);
			offset = outlen - GZIP_CONTEXT(context)->z.avail_out;
		} while( r == Z_OK && GZIP_CONTEXT(context)->z.avail_in != 0 );
	} else {
		do {
			size_t newlen = outlen * 2;
			if (newlen > outlen)
				outlen = newlen;
			else
				abort();	/* Someone has somehow malloced over 2^64 bits of ram. */
			
			out = realloc(out, outlen);
			if( out == NULL ) abort();

			GZIP_CONTEXT(context)->z.next_out = ((unsigned char *)out) + offset;
			GZIP_CONTEXT(context)->z.avail_out = outlen - offset;

			r = deflate(&GZIP_CONTEXT(context)->z, Z_FINISH);
			offset = outlen - GZIP_CONTEXT(context)->z.avail_out;
		} while( r == Z_OK && r != Z_STREAM_END /* no-op */);
	}

	if( (r != Z_OK && r != Z_STREAM_END) ) {
		xar_err_new(x);
		xar_err_set_file(x, f);
		xar_err_set_string(x, "Error compressing file");
		xar_err_set_errno(x, r);
		xar_err_callback(x, XAR_SEVERITY_FATAL, XAR_ERR_ARCHIVE_CREATION);
		return -1;
	}

	free(*in);
	*in = out;
	GZIP_CONTEXT(context)->count += *inlen;
	*inlen = offset;
	return 0;
}
