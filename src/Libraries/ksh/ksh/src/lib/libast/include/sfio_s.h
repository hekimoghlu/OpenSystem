/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 25, 2023.
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
#ifndef _SFIO_S_H
#define _SFIO_S_H	1

/*
 * sfio file structure used by sfio and the stdio source compatibility library
 */

#if !defined(_SFHDR_H) && defined(_SFIO_H) && SFIO_VERSION < 20020214L
#define	_data		data
#define	_endb		endb
#define	_next		next
#endif

struct _sfio_s
{	unsigned char*	_next;	/* next position to read/write from	*/
	unsigned char*	_endw;	/* end of write buffer			*/
	unsigned char*	_endr;	/* end of read buffer			*/
	unsigned char*	_endb;	/* end of buffer			*/
	struct _sfio_s*	_push;	/* the stream that was pushed on	*/
	unsigned short	_flags;	/* type of stream			*/
	short		_file;	/* file descriptor			*/
	unsigned char*	_data;	/* base of data buffer			*/
	ssize_t		_size;	/* buffer size				*/
	ssize_t		_val;	/* values or string lengths		*/
#ifdef _SFIO_PRIVATE
	_SFIO_PRIVATE
#endif
};

#endif
