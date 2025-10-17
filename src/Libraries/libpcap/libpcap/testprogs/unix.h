/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 23, 2022.
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
#ifndef unix_h
#define	unix_h

/*
 * Definitions to make MSVC C runtime library structures and functions
 * look like the UNIX structures and functions they are intended to
 * resemble.
 */
#ifdef _MSC_VER
  #define stat		_stat
  #define fstat		_fstat

  #define open		_open
  #define O_RDONLY	_O_RDONLY
  #define O_WRONLY	_O_WRONLY
  #define O_RDWR	_O_RDWR
  #define O_BINARY	_O_BINARY
  #define O_CREAT	_O_CREAT
  #define O_TRUNC	_O_TRUNC
  #define read		_read
  #define write		_write
  #define close		_close
#endif

#endif
