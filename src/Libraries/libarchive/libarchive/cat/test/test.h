/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 24, 2024.
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
/* Every test program should #include "test.h" as the first thing. */

#define KNOWNREF       "test_expand.Z.uu"
#define ENVBASE "BSDCAT"  /* Prefix for environment variables. */
#define	PROGRAM "bsdcat"  /* Name of program being tested. */
#define PROGRAM_ALIAS "cat" /* Generic alias for program */
#undef	LIBRARY		  /* Not testing a library. */
#undef	EXTRA_DUMP	  /* How to dump extra data */
#undef	EXTRA_ERRNO	  /* How to dump errno */
/* How to generate extra version info. */
#define	EXTRA_VERSION    (systemf("%s --version", testprog) ? "" : "")

#include "test_common.h"
