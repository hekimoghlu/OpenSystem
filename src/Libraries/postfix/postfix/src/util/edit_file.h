/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 3, 2025.
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

#ifndef _EDIT_FILE_H_INCLUDED_
#define _EDIT_FILE_H_INCLUDED_

/*++
/* NAME
/*	edit_file 3h
/* SUMMARY
/*	simple cooperative file updating protocol
/* SYNOPSIS
/*	#include <edit_file.h>
/* DESCRIPTION
/* .nf

 /*
  * Utility library.
  */
#include <vstream.h>

 /*
  * External interface.
  */
typedef struct {
    /* Private. */
    char   *final_path;
    mode_t  final_mode;
    /* Public. */
    char   *tmp_path;
    VSTREAM *tmp_fp;
} EDIT_FILE;

#define EDIT_FILE_SUFFIX ".tmp"

extern EDIT_FILE *edit_file_open(const char *, int, mode_t);
extern int WARN_UNUSED_RESULT edit_file_close(EDIT_FILE *);
extern void edit_file_cleanup(EDIT_FILE *);

/* LICENSE
/* .ad
/* .fi
/*	The Secure Mailer license must be distributed with this software.
/* AUTHOR(S)
/*	Wietse Venema
/*	IBM T.J. Watson Research
/*	P.O. Box 704
/*	Yorktown Heights, NY 10598, USA
/*--*/

#endif
