/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 6, 2024.
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
/* $Id: backtrace-emptytbl.c,v 1.3 2009/09/01 20:13:44 each Exp $ */

/*! \file */

/*
 * This file defines an empty (default) symbol table used in backtrace.c
 * If the application wants to have a complete symbol table, it should redefine
 * isc__backtrace_symtable with the complete table in some way, and link the
 * version of the library not including this definition
 * (e.g. libisc-nosymbol.a).
 */

#include <config.h>

#include <isc/backtrace.h>

LIBISC_EXTERNAL_DATA const int isc__backtrace_nsymbols = 0;
LIBISC_EXTERNAL_DATA const
	isc_backtrace_symmap_t isc__backtrace_symtable[] = { { NULL, "" } };
