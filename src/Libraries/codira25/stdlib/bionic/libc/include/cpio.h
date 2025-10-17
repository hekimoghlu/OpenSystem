/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 29, 2022.
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
 * @file cpio.h
 * @brief Constants for reading/writing cpio files.
 */

#include <sys/cdefs.h>

/** Readable by user mode field bit. */
#define C_IRUSR 0000400
/** Writable by user mode field bit. */
#define C_IWUSR 0000200
/** Executable by user mode field bit. */
#define C_IXUSR 0000100
/** Readable by group mode field bit. */
#define C_IRGRP 0000040
/** Writable by group mode field bit. */
#define C_IWGRP 0000020
/** Executable by group mode field bit. */
#define C_IXGRP 0000010
/** Readable by other mode field bit. */
#define C_IROTH 0000004
/** Writable by other mode field bit. */
#define C_IWOTH 0000002
/** Executable by other mode field bit. */
#define C_IXOTH 0000001
/** Set-UID mode field bit. */
#define C_ISUID 0004000
/** Set-GID mode field bit. */
#define C_ISGID 0002000
/** Directory restricted deletion mode field bit. */
#define C_ISVTX 0001000
/** Directory mode field type. */
#define C_ISDIR 0040000
/** FIFO mode field type. */
#define C_ISFIFO 0010000
/** Regular file mode field type. */
#define C_ISREG 0100000
/** Block special file mode field type. */
#define C_ISBLK 0060000
/** Character special file mode field type. */
#define C_ISCHR 0020000
/** Reserved. */
#define C_ISCTG 0110000
/** Symbolic link mode field type. */
#define C_ISLNK 0120000
/** Socket mode field type. */
#define C_ISSOCK 0140000

/** cpio file magic. */
#define MAGIC "070707"
