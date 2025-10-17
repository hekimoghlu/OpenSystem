/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 29, 2022.
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

#include "arcmod.h"
#include "archive.h"
#include "stat.h"
#include "data.h"
#include "linuxattr.h"
#include "fbsdattr.h"
#include "darwinattr.h"
#include "ext2.h"
#include "xar.h"
#include <string.h>

struct arcmod xar_arcmods[] = {
	{ xar_stat_archive, xar_stat_extract },      /* must be first */
	{ xar_linuxattr_archive, xar_linuxattr_extract },
	{ xar_fbsdattr_archive, xar_fbsdattr_extract },
	{ xar_darwinattr_archive, xar_darwinattr_extract },
	{ xar_ext2attr_archive, xar_ext2attr_extract },
	{ xar_data_archive, xar_data_extract },
	/* Add new modules here */
	{ NULL, xar_set_perm },
	{ NULL, xar_flags_extract }
};

/* xar_arcmod_archive
 * x: archive to add the file to
 * f: node representing the file
 * file: the filesystem path to the file
 * Returns: 0 on success
 * Summary: This is the entry point to actual file archival.
 */
int32_t xar_arcmod_archive(xar_t x, xar_file_t f, const char *file, const char *buffer, size_t len) {
	int i;
	int32_t ret;
	for(i = 0; i < (sizeof(xar_arcmods)/sizeof(struct arcmod)); i++) {
		if( xar_arcmods[i].archive ) {
			ret = xar_arcmods[i].archive(x, f, file, buffer, len);
			if( ret < 0 ) {
				return ret;
			}
			if( ret > 0 ) {
				return 0;
			}
		}
	}
	return 0;
}

/* xar_arcmod_extract
 * x: archive to extract the file from
 * f: node representing the file
 * file: the filesystem path to the target file
 * Returns: 0 on success
 * Summary: This is the entry point to actual file archival.
 */
int32_t xar_arcmod_extract(xar_t x, xar_file_t f, const char *file, char *buffer, size_t len) {
	int i;
	int32_t ret;
	for(i = 0; i < (sizeof(xar_arcmods)/sizeof(struct arcmod)); i++) {
		if( xar_arcmods[i].extract ) {
			ret = xar_arcmods[i].extract(x, f, file, buffer, len);
			if( ret < 0 ) {
				// If the extract failed we have corrupt asset on disk, remove, well try.
				unlink(file);
				return ret;
			}
			if( ret > 0 ) {
				return 0;
			}
		}
	}
	return 0;
}


int32_t xar_arcmod_verify(xar_t x, xar_file_t f, xar_progress_callback p){
	return xar_data_verify(x,f, p);
}

/* xar_check_prop
 * x: xar archive
 * name: name of property to check
 * Description: If XAR_OPT_PROPINCLUDE is set at all, only properties
 * specified for inclusion will be added.
 * If XAR_OPT_PROPINCLUDE is not set, and XAR_OPT_PROPEXCLUDE is set,
 * properies specified by XAR_OPT_PROPEXCLUDE will be omitted.
 * Returns: 0 for not to include, 1 for include.
 */
int32_t xar_check_prop(xar_t x, const char *name) {
	xar_attr_t i;
	char includeset = 0;

	for(i = XAR(x)->attrs; i; i = XAR_ATTR(i)->next) {
		if( strcmp(XAR_ATTR(i)->key, XAR_OPT_PROPINCLUDE) == 0 ) {
			if( strcmp(XAR_ATTR(i)->value, name) == 0 )
				return 1;
			includeset = 1;
		}
	}

	if( includeset )
		return 0;

	for(i = XAR(x)->attrs; i; i = XAR_ATTR(i)->next) {
		if( strcmp(XAR_ATTR(i)->key, XAR_OPT_PROPEXCLUDE) == 0 ) {
			if( strcmp(XAR_ATTR(i)->value, name) == 0 )
				return 0;
		}
	}

	return 1;
}
