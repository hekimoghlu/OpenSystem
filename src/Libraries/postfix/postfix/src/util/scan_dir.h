/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 31, 2023.
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

#ifndef _SCAN_DIR_H_INCLUDED_
#define _SCAN_DIR_H_INCLUDED_

/*++
/* NAME
/*	scan_dir 3h
/* SUMMARY
/*	directory scanner
/* SYNOPSIS
/*	#include <scan_dir.h>
/* DESCRIPTION
/* .nf

 /*
  * The directory scanner interface.
  */
typedef struct SCAN_DIR SCAN_DIR;

extern SCAN_DIR *scan_dir_open(const char *);
extern char *scan_dir_next(SCAN_DIR *);
extern char *scan_dir_path(SCAN_DIR *);
extern void scan_dir_push(SCAN_DIR *, const char *);
extern SCAN_DIR *scan_dir_pop(SCAN_DIR *);
extern SCAN_DIR *scan_dir_close(SCAN_DIR *);

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
