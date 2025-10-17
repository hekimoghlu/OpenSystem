/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 3, 2024.
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
#pragma once

/**
 * @file ar.h
 * @brief Constants for reading/writing `.a` files.
 */

#include <sys/cdefs.h>

/** The magic at the beginning of a `.a` file. */
#define ARMAG  "!<arch>\n"
/** The length of the magic at the beginning of a `.a` file. */
#define SARMAG  8

/** The contents of every `ar_hdr::ar_fmag` field.*/
#define	ARFMAG	"`\n"

struct ar_hdr {
  /* Name. */
  char ar_name[16];
  /* Modification time. */
  char ar_date[12];
  /** User id. */
  char ar_uid[6];
  /** Group id. */
  char ar_gid[6];
  /** Octal file permissions. */
  char ar_mode[8];
  /** Size in bytes. */
  char ar_size[10];
  /** Consistency check. Always contains `ARFMAG`. */
  char ar_fmag[2];
};
