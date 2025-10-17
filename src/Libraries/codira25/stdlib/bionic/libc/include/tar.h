/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 14, 2022.
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
 * @file tar.h
 * @brief Constants for reading/writing `.tar` files.
 */

#include <sys/cdefs.h>

/** `.tar` file magic. (Includes the NUL.) */
#define TMAGIC "ustar"
/** `.tar` file magic length in bytes. */
#define TMAGLEN 6
/** `.tar` file version. (Does not include the NUL.) */
#define TVERSION "00"
/** `.tar` file version length in bytes. */
#define TVERSLEN 2

/** Regular file type flag. */
#define REGTYPE '0'
/** Alternate regular file type flag. */
#define AREGTYPE '\0'
/** Link type flag. */
#define LNKTYPE '1'
/** Symbolic link type flag. */
#define SYMTYPE '2'
/** Character special file type flag. */
#define CHRTYPE '3'
/** Block special file type flag. */
#define BLKTYPE '4'
/** Directory type flag. */
#define DIRTYPE '5'
/** FIFO special file type flag. */
#define FIFOTYPE '6'
/** Reserved type flag. */
#define CONTTYPE '7'

/** Set-UID mode field bit. */
#define TSUID 04000
/** Set-GID mode field bit. */
#define TSGID 02000
/** Directory restricted deletion mode field bit. */
#define TSVTX 01000
/** Readable by user mode field bit. */
#define TUREAD 00400
/** Writable by user mode field bit. */
#define TUWRITE 00200
/** Executable by user mode field bit. */
#define TUEXEC 00100
/** Readable by group mode field bit. */
#define TGREAD 00040
/** Writable by group mode field bit. */
#define TGWRITE 00020
/** Executable by group mode field bit. */
#define TGEXEC 00010
/** Readable by other mode field bit. */
#define TOREAD 00004
/** Writable by other mode field bit. */
#define TOWRITE 00002
/** Executable by other mode field bit. */
#define TOEXEC 00001
