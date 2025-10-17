/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 12, 2023.
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

#ifndef _DICT_LMDB_H_INCLUDED_
#define _DICT_LMDB_H_INCLUDED_

/*++
/* NAME
/*	dict_lmdb 3h
/* SUMMARY
/*	dictionary manager interface to OpenLDAP LMDB files
/* SYNOPSIS
/*	#include <dict_lmdb.h>
/* DESCRIPTION
/* .nf

 /*
  * Utility library.
  */
#include <dict.h>

 /*
  * External interface.
  */
#define DICT_TYPE_LMDB	"lmdb"

extern DICT *dict_lmdb_open(const char *, int, int);

 /*
  * XXX Should be part of the DICT interface.
  */
extern size_t dict_lmdb_map_size;

 /* Minimum size without SIGSEGV. */
#define DEFINE_DICT_LMDB_MAP_SIZE size_t dict_lmdb_map_size = 8192

/* LICENSE
/* .ad
/* .fi
/*	The Secure Mailer license must be distributed with this software.
/* AUTHOR(S)
/*	Howard Chu
/*	Symas Corporation
/*--*/

#endif
