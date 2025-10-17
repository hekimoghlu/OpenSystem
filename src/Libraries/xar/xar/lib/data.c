/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 25, 2024.
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
 * Portions Copyright 2006, Apple Computer, Inc.
 * Christopher Ryan <ryanc@apple.com>
*/

#define _FILE_OFFSET_BITS 64
#include <stdlib.h>
#include <stdio.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <limits.h>
#include <unistd.h>
#include <inttypes.h>
#include <sys/types.h>

#include "xar.h"
#include "filetree.h"
#include "archive.h"
#include "io.h"
#include "arcmod.h"
#include "data.h"

#ifndef O_EXLOCK
#define O_EXLOCK 0
#endif



int32_t xar_data_read(xar_t x, xar_file_t f, void *inbuf, size_t bsize, void *context) {
	int32_t r;

	/* read from buffer, rather then fd,if available */
	if(DATA_CONTEXT(context)->length){
		char *readbuf = (char *)DATA_CONTEXT(context)->buffer;
		size_t sizetoread = DATA_CONTEXT(context)->length - DATA_CONTEXT(context)->offset;
		
		if( !sizetoread){
			return 0;
		}
		
		if( sizetoread > bsize ){
			sizetoread = bsize;
		}
		
		/* dont read passed the end of the buffer */
		if((DATA_CONTEXT(context)->offset + sizetoread) > DATA_CONTEXT(context)->length){
			return -1;
			//sizetoread = (DATA_CONTEXT(context)->offset + sizetoread) - DATA_CONTEXT(context)->length;
		}
		
		readbuf += DATA_CONTEXT(context)->offset;
		memcpy(inbuf,readbuf,sizetoread);
		
		DATA_CONTEXT(context)->total += sizetoread;
		DATA_CONTEXT(context)->offset += sizetoread;
		
		return sizetoread;
	}
	
	while(1) {
		r = read(DATA_CONTEXT(context)->fd, inbuf, bsize);
		if( (r < 0) && (errno == EINTR) )
			continue;
		DATA_CONTEXT(context)->total += r;
		return r;
	}		
	
}

int32_t xar_data_write(xar_t x, xar_file_t f, void *buf, size_t len, void *context) {
	int32_t r;
	size_t off = 0;
	
	/* read from buffer, rather then fd,if available */
	if(DATA_CONTEXT(context)->length){
		char *writebuf = (char *)DATA_CONTEXT(context)->buffer;
		
		/* dont write passed the end of the buffer */
		if((DATA_CONTEXT(context)->offset + len) > DATA_CONTEXT(context)->length){
			return -1;
		}
		
		writebuf += DATA_CONTEXT(context)->offset;
		memcpy(writebuf,buf,len);
		
		DATA_CONTEXT(context)->offset += len;
		
		return len;
	}
	
	do {
		r = write(DATA_CONTEXT(context)->fd, ((char *)buf)+off, len-off);
		if( (r < 0) && (errno != EINTR) )
			return r;
		off += r;
	} while( off < len );
	return off;
}

/* xar_data_archive
 * This is the arcmod archival entry point for archiving the file's
 * data into the heap file.
 */
int32_t xar_data_archive(xar_t x, xar_file_t f, const char *file, const char *buffer, size_t len) {
	const char *opt;
	int32_t retval = 0;
	struct _data_context context;
	xar_prop_t tmpp;
	
	memset(&context,0,sizeof(struct _data_context));

	if( !xar_check_prop(x, "data") )
		return 0;

	xar_prop_get(f, "type", &opt);
	if(!opt) return 0;
	if( strcmp(opt, "file") != 0 ) {
		if( strcmp(opt, "hardlink") == 0 ) {
			opt = xar_attr_get(f, "type", "link");
			if( !opt )
				return 0;
			if( strcmp(opt, "original") != 0 )
				return 0;
			/* else, we're an original hardlink, so keep going */
		} else
			return 0;
	}

	if( 0 == len ){
		context.fd = open(file, O_RDONLY);
		if( context.fd < 0 ) {
			xar_err_new(x);
			xar_err_set_file(x, f);
			xar_err_set_string(x, "io: Could not open file");
			xar_err_callback(x, XAR_SEVERITY_NONFATAL, XAR_ERR_ARCHIVE_CREATION);
			return -1;
		}		
	}else{
		context.buffer = (void *)buffer;
		context.length = len;
		context.offset = 0;
	}

#ifdef F_NOCACHE
	fcntl(context.fd, F_NOCACHE, 1);
#endif

	tmpp = xar_prop_pset(f, NULL, "data", NULL);
	retval = XAR(x)->attrcopy_to_heap(x, f, tmpp, xar_data_read,(void *)(&context));
	if( context.total == 0 )
		xar_prop_unset(f, "data");

	if(context.fd > 0){
		close(context.fd);
		context.fd = -1;
	}
	
	return retval;
}

int32_t xar_data_extract(xar_t x, xar_file_t f, const char *file, char *buffer, size_t len) {
	const char *opt;
	int32_t retval = 0;
	struct _data_context context;
	xar_prop_t tmpp;
	
	memset(&context,0,sizeof(struct _data_context));
	
	/* Only regular files are copied in and out of the heap here */
	xar_prop_get(f, "type", &opt);
	if( !opt ) return 0;
	if( strcmp(opt, "file") != 0 ) {
		if( strcmp(opt, "hardlink") == 0 ) {
			opt = xar_attr_get(f, "type", "link");
			if( !opt )
				return 0;
			if( strcmp(opt, "original") != 0 )
				return 0; 
			/* else, we're an original hardlink, so keep going */
		} else
			return 0;
	}
	
	if ( len ){
		context.length = len;
		context.buffer = buffer;
		context.offset = 0;
	}else{
		/* mode 600 since other modules may need to operate on the file
		* prior to the real permissions being set.
		*/
		context.fd = open(file, O_RDWR|O_TRUNC|O_EXLOCK, 0600);
		if( context.fd < 0 ) {
			xar_err_new(x);
			xar_err_set_file(x, f);
			xar_err_set_string(x, "io: Could not create file");
			xar_err_callback(x, XAR_SEVERITY_NONFATAL, XAR_ERR_ARCHIVE_EXTRACTION);
			return -1;
		}
		
	}
	
	tmpp = xar_prop_pfirst(f);
	if( tmpp )
		tmpp = xar_prop_find(tmpp, "data");
	if( !tmpp ) {
		close(context.fd);
		return 0;
	}
	retval = XAR(x)->attrcopy_from_heap(x, f, tmpp, xar_data_write, (void *)(&context));
	
	if( context.fd > 0 ){		
		close(context.fd);
		context.fd = -1;
	}
	
	return retval;
}

int32_t xar_data_verify(xar_t x, xar_file_t f, xar_progress_callback p)
{
	const char *opt;
	struct _data_context context;
	xar_prop_t tmpp;
	
	memset(&context,0,sizeof(struct _data_context));
	
	context.progress = p;

	/* Only regular files are copied in and out of the heap here */
	xar_prop_get(f, "type", &opt);
	if( !opt ) return 0;
	if( strcmp(opt, "directory") == 0 ) {
		return 0;
	}
	
	tmpp = xar_prop_pfirst(f);
	if( tmpp )
		tmpp = xar_prop_find(tmpp, "data");
	
	if (!tmpp)		// It appears that xar can have truely empty files, aka, no data. We should just fail to verify these files. 
		return 0;	// After all, the checksum of blank is meaningless. So, failing to do so will cause a crash.
	
	return XAR(x)->attrcopy_from_heap(x, f, tmpp, NULL , (void *)(&context));
}
