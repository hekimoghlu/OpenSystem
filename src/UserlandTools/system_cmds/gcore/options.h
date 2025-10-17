/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 2, 2022.
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
#include <sys/types.h>
#include <compression.h>
#include <stdbool.h>

#include <assert.h>

#ifndef _OPTIONS_H
#define _OPTIONS_H

#if defined(__arm__) || defined(__arm64__)
#define RDAR_23744374       1   /* 'true' while not fixed i.e. enable workarounds */
#define RDAR_28040018		1	/* 'true' while not fixed i.e. enable workarounds */
#endif

#define CONFIG_SUBMAP       1   /* include submaps */
#define CONFIG_GCORE_MAP	1	/* support 'gcore map' */
#define CONFIG_GCORE_CONV	1	/* support 'gcore conv' - new -> old core files */
#define CONFIG_GCORE_FREF	1	/* support 'gcore fref' - referenced file list */
#define CONFIG_DEBUG		1	/* support '-d' option */

#ifdef NDEBUG
#define poison(a, p, s)     /* do nothing */
#else
#define poison(a, p, s)     memset(a, p, s) /* scribble on dying memory */
#endif

struct options {
    int corpsify;       // make a corpse to dump from
    int suspend;        // suspend while dumping
    int preserve;       // preserve the core file, even if there are errors
    int verbose;        // be chatty
#ifdef CONFIG_DEBUG
    int debug;          // internal debugging: options accumulate. noisy.
#endif
	int extended;		// avoid writing out ro mapped files, compress regions
    int skinny;         // just code segments, mapped files as references, mutually exclusive with extended
    off_t sizebound;    // maximum size of the dump
    size_t chunksize;   // max size of a compressed subregion
    compression_algorithm calgorithm; // algorithm in use
	size_t ncthresh;	// F_NOCACHE enabled *above* this value
    int allfilerefs;    // if set, every mapped file on the root fs is a fileref
	int dsymforuuid;    // Try dsysForUUID to retrieve symbol-rich executable
    int gzip;           // pipe corefile via gzip -1 compression
    int stream;         // write corefile sequentially e.g. no pwrites
    bool notes;         // if set, dump LC_NOTES for memory analysis tools
};

extern const struct options *opt;

/*
 * == 0 - not verbose
 * >= 1 - verbose plus chatty
 * >= 2 - tabular summaries
 * >= 3 - all
 */

#ifdef CONFIG_DEBUG
#define OPTIONS_DEBUG(opt, lvl)	((opt)->debug && (opt)->debug >= (lvl))
#else
#define OPTIONS_DEBUG(opt, lvl)	0
#endif

#endif /* _OPTIONS_H */
